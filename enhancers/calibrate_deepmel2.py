import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from cgm.cgm_distribution import (  # noqa: E402
    PCATransform,
    _mean_off_diagonal_kernel,
    _mmd_calibration_terms,
    dot_product_kernel,
    energy_distance_kernel,
    fit_pca,
    scale_kernel,
)
from cgm.lr_schedules import make_warmup_cosine_scheduler  # noqa: E402
from simple_diff.model import CNNConditionalMDLM  # noqa: E402

DEFAULT_CACHE_PATH = Path("enhancers/cache/deepmel2_alphagenome_features.pt")
AUTO_SCALE_CONDITIONS = 5
AUTO_SCALE_SEED = 0
AUTO_SCALE_TARGET_CONSTRAINT = 1.0
AUTOCAST_DTYPES = {
    "none": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class DiffusionSample:
    intermediates: torch.Tensor
    next_unmasks: torch.Tensor
    cond: torch.Tensor
    log_p_theta: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        return self.intermediates[:, -1]


@dataclass
class StandardizeTransform:
    mu: torch.Tensor
    sigma: torch.Tensor

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mu) / self.sigma


CalibrationTransform = PCATransform | StandardizeTransform


@dataclass
class SplitCache:
    raw_features: torch.Tensor
    log_features: torch.Tensor
    calibration_features: torch.Tensor
    condition_matrix: torch.Tensor
    condition_to_indices: dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conditionally calibrate a pretrained DeepMEL2 masked diffusion model "
            "against AlphaGenome feature distributions."
        )
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--target-cache-pt", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final learning rate as a fraction of --lr for cosine scheduling.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument(
        "--kernel",
        choices=["energy", "dotproduct"],
        default="energy",
        help="Kernel for the MMD calibration objective.",
    )
    parser.add_argument(
        "--no-loo",
        "--no_loo",
        dest="no_loo",
        action="store_true",
        help=(
            "Disable leave-one-out correction for the MMD coefficients. "
            "KL centering is still applied."
        ),
    )
    parser.add_argument(
        "--loss-weighting",
        choices=["raw", "normalized"],
        default="normalized",
        help="How to combine KL and MMD coefficients.",
    )
    parser.add_argument(
        "--feature-transform",
        choices=["pca", "standardize"],
        default="pca",
        help="Feature transform applied to log1p AlphaGenome features before MMD.",
    )
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--pca-whiten", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-autocast-dtype",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="Optional autocast dtype around diffusion model forward passes.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--alphagenome-model-path", type=str, default=None)
    parser.add_argument("--background-file", type=str, default=None)
    parser.add_argument(
        "--alphagenome-autocast-dtype",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="Optional autocast dtype around AlphaGenome model.predict calls.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default=None,
        help="Optional torch float32 matmul precision setting.",
    )
    parser.add_argument("--final-samples-per-condition", type=int, default=16)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--wandb-project", type=str, default="kcgm_enhancer_calibration"
    )
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def condition_key(condition_idx: int) -> str:
    return f"condition_{condition_idx}"


def build_n_unmask_per_step(seq_len: int, sample_steps: int) -> torch.Tensor:
    if sample_steps < 1:
        raise ValueError("--sample-steps must be positive.")
    t = torch.linspace(1, 0, sample_steps + 1)
    alpha_t = 1 - torch.cos(torch.pi / 2 * (1 - t))
    n_masked = torch.round((1 - alpha_t) * seq_len)
    num_to_unmask = n_masked[:-1] - n_masked[1:]
    num_to_unmask = num_to_unmask[num_to_unmask != 0].to(torch.long)
    if int(num_to_unmask.sum()) != seq_len:
        raise RuntimeError("Internal unmask schedule does not cover the full sequence.")
    if (num_to_unmask <= 0).any():
        raise RuntimeError("Internal unmask schedule has nonpositive steps.")
    return num_to_unmask


class TauLeapingDiffusionPolicy:
    def __init__(
        self,
        model: CNNConditionalMDLM,
        *,
        seq_len: int,
        sample_steps: int,
        temperature: float,
        autocast_dtype: str,
        device: torch.device,
    ):
        self.model = torch.compile(model, mode="reduce-overhead")
        self.seq_len = seq_len
        self.temperature = temperature
        self.autocast_dtype = AUTOCAST_DTYPES[autocast_dtype]
        self.device = device
        self.mask_token = model.mask_token
        self.num_to_unmask = build_n_unmask_per_step(seq_len, sample_steps).to(device)

    @property
    def num_sample_chunks(self) -> int:
        return len(self.num_to_unmask)

    def _get_logits(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype or torch.bfloat16,
            enabled=self.autocast_dtype is not None,
        ):
            logits = self.model(x, cond=cond) / self.temperature
        return logits.float()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, verbose: bool = False) -> DiffusionSample:
        cond = cond.to(self.device).float()
        bsz = cond.shape[0]
        x = torch.full((bsz, self.seq_len), self.mask_token, device=self.device)
        next_pos_probs = torch.ones_like(x, dtype=torch.float32)
        batch_range = torch.arange(bsz, device=self.device)
        seq_log_probs = torch.zeros(bsz, device=self.device)

        intermediates = [x.clone().cpu()]
        next_unmasks = []

        for num_to_unmask in tqdm(
            self.num_to_unmask,
            desc="Tau-leap sampling",
            disable=not verbose,
            leave=False,
        ):
            k = int(num_to_unmask)
            logits = self._get_logits(x, cond)
            next_unmask_idx = torch.multinomial(
                next_pos_probs,
                num_samples=k,
                replacement=False,
            )

            chosen_mask = torch.zeros_like(x, dtype=torch.bool)
            chosen_mask.scatter_(1, next_unmask_idx, True)
            next_unmasks.append(chosen_mask.cpu())

            next_pos_probs.scatter_(
                1,
                next_unmask_idx,
                torch.zeros_like(next_unmask_idx, dtype=next_pos_probs.dtype),
            )

            token_logits = logits.gather(
                1,
                next_unmask_idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1]),
            ).float()
            token_dist = Categorical(logits=token_logits)
            token_vals = token_dist.sample()
            x[batch_range.unsqueeze(1), next_unmask_idx] = token_vals
            seq_log_probs += token_dist.log_prob(token_vals).sum(dim=1)
            intermediates.append(x.clone().cpu())

        return DiffusionSample(
            intermediates=torch.stack(intermediates, dim=1),
            next_unmasks=torch.stack(next_unmasks, dim=1),
            cond=cond.detach().cpu(),
            log_p_theta=seq_log_probs.detach().cpu(),
        )

    def log_p(
        self,
        sample: DiffusionSample,
        *,
        sample_idx: int,
        batch_idx: int = 0,
        batch_chunks: int = 1,
    ) -> torch.Tensor:
        bsz = sample.intermediates.shape[0]
        if bsz % batch_chunks != 0:
            raise ValueError(f"Batch size {bsz} is not divisible by {batch_chunks}.")
        step = bsz // batch_chunks
        batch_start = batch_idx * step
        batch_end = batch_start + step

        x = sample.intermediates[batch_start:batch_end, sample_idx].to(self.device)
        y_next = sample.intermediates[batch_start:batch_end, sample_idx + 1].to(
            self.device
        )
        mask = sample.next_unmasks[batch_start:batch_end, sample_idx].to(self.device)
        cond = sample.cond[batch_start:batch_end].to(self.device).float()
        y_next = torch.where(mask, y_next, torch.zeros_like(y_next))

        logits = self._get_logits(x, cond)
        all_log_probs = logits.log_softmax(dim=-1)
        pos_log_probs = all_log_probs.gather(
            dim=-1,
            index=y_next.unsqueeze(-1),
        ).squeeze(-1)
        return (pos_log_probs * mask).sum(dim=1)

    @torch.no_grad()
    def log_p_total(self, sample: DiffusionSample) -> torch.Tensor:
        return torch.stack(
            [
                self.log_p(sample, sample_idx=sample_idx)
                for sample_idx in range(self.num_sample_chunks)
            ],
            dim=0,
        ).sum(dim=0)


def load_model(path: Path, device: torch.device) -> CNNConditionalMDLM:
    model = CNNConditionalMDLM.load_from_checkpoint(str(path), map_location=device)
    return model.to(device)


def freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def fit_feature_transform(args: argparse.Namespace, train_log_features: torch.Tensor):
    if args.feature_transform == "pca":
        return fit_pca(
            train_log_features,
            n_components=args.pca_components,
            whiten=args.pca_whiten,
        )

    sigma = train_log_features.std(dim=0).clamp_min(1e-6)
    return StandardizeTransform(mu=train_log_features.mean(dim=0), sigma=sigma)


def to_device_transform(
    transform: CalibrationTransform,
    device: torch.device,
) -> CalibrationTransform:
    if isinstance(transform, PCATransform):
        return PCATransform(
            mu=transform.mu.to(device),
            evecs=transform.evecs.to(device),
            scales=transform.scales.to(device),
        )
    return StandardizeTransform(
        mu=transform.mu.to(device),
        sigma=transform.sigma.to(device),
    )


def serialize_feature_transform(
    transform: CalibrationTransform,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if isinstance(transform, PCATransform):
        return {
            "kind": "pca",
            "mu": transform.mu,
            "evecs": transform.evecs,
            "scales": transform.scales,
            "pca_components": args.pca_components,
            "pca_whiten": args.pca_whiten,
        }
    return {
        "kind": "standardize",
        "mu": transform.mu,
        "sigma": transform.sigma,
    }


def load_split_cache(
    payload: dict[str, Any],
    split: str,
    feature_transform: CalibrationTransform,
) -> SplitCache:
    split_payload = payload["splits"][split]
    raw_features = split_payload["features"].to(torch.float32)
    log_features = torch.log1p(raw_features)
    calibration_features = feature_transform(log_features)
    return SplitCache(
        raw_features=raw_features,
        log_features=log_features,
        calibration_features=calibration_features,
        condition_matrix=split_payload["condition_matrix"].to(torch.float32),
        condition_to_indices={
            key: value.to(torch.long)
            for key, value in split_payload["condition_to_indices"].items()
        },
    )


def valid_condition_indices(
    split_cache: SplitCache,
    *,
    min_count: int,
) -> list[int]:
    condition_dim = split_cache.condition_matrix.shape[1]
    valid = []
    for condition_idx in range(condition_dim):
        key = condition_key(condition_idx)
        if key in split_cache.condition_to_indices:
            if split_cache.condition_to_indices[key].numel() >= min_count:
                valid.append(condition_idx)
    return valid


def sample_condition_matrix(
    split_cache: SplitCache,
    condition_idx: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    idx = split_cache.condition_to_indices[condition_key(condition_idx)]
    draws = torch.randint(idx.numel(), (batch_size,), generator=generator)
    return split_cache.condition_matrix[idx[draws]]


def make_feature_extractor(args: argparse.Namespace):
    from features import build_feature_extractor

    kwargs = {
        "batch_size": args.feature_batch_size,
        "autocast_dtype": args.alphagenome_autocast_dtype,
        "matmul_precision": args.matmul_precision,
        "verbose": False,
    }
    if args.alphagenome_model_path is not None:
        kwargs["model_path"] = args.alphagenome_model_path
    if args.background_file is not None:
        kwargs["bg_file"] = args.background_file
    return build_feature_extractor(**kwargs)


def transform_generated_features(
    raw_features: torch.Tensor,
    feature_transform: CalibrationTransform,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_features = raw_features.to(device=device, dtype=torch.float32)
    log_features = torch.log1p(raw_features)
    return log_features, feature_transform(log_features)


def compute_kl_coefficients(
    log_p_theta: torch.Tensor,
    log_p_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kls = log_p_theta - log_p_base
    kl_loss_val = kls.mean()
    if kls.numel() > 1:
        kls = kls - (kls.sum() - kls) / (kls.numel() - 1)
    return kls / kls.numel(), kl_loss_val


def combine_coefficients(
    c_kl: torch.Tensor,
    c_mmd: torch.Tensor,
    kl_loss_val: torch.Tensor,
    mmd_estimate: torch.Tensor,
    lambd: float,
    loss_weighting: str,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    if loss_weighting == "raw":
        kl_weight = lambd
        constraint_weight = 1.0
    else:
        denom = 1.0 + lambd
        kl_weight = lambd / denom
        constraint_weight = 1.0 / denom
    c_total = kl_weight * c_kl + constraint_weight * c_mmd
    total_loss_val = kl_weight * kl_loss_val + constraint_weight * mmd_estimate
    return c_total, total_loss_val, kl_weight, constraint_weight


def abs_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x_abs = x.detach().abs()
    return {
        f"{prefix}_abs_mean": x_abs.mean().item(),
        f"{prefix}_abs_max": x_abs.max().item(),
    }


def prefixed_abs_stats(prefix: str, name: str, x: torch.Tensor) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in abs_stats(name, x).items()}


def mean_metrics(metrics: list[dict[str, float]], prefix: str) -> dict[str, float]:
    numeric_keys = [
        key
        for key, value in metrics[0].items()
        if isinstance(value, int | float) and key != f"{prefix}/condition_idx"
    ]
    return {
        key.replace(f"{prefix}/", "epoch/"): sum(metric[key] for metric in metrics)
        / len(metrics)
        for key in numeric_keys
    }


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().pow(2).sum().item()
    return total**0.5


def estimate_condition_mmd(
    generated_features: torch.Tensor,
    target_features: torch.Tensor,
    kernel,
) -> torch.Tensor:
    if generated_features.shape[0] < 2 or target_features.shape[0] < 2:
        return torch.full((), float("nan"), device=generated_features.device)
    kxx = kernel(generated_features, generated_features)
    kxy = kernel(generated_features, target_features)
    kyy = kernel(target_features, target_features)
    n = generated_features.shape[0]
    m = target_features.shape[0]
    kxx_mean = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    kyy_mean = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    return kxx_mean - 2 * kxy.mean() + kyy_mean


@torch.no_grad()
def final_samples_and_metrics(
    policy: TauLeapingDiffusionPolicy,
    base: TauLeapingDiffusionPolicy,
    valid_cache: SplitCache,
    feature_extractor,
    feature_transform: CalibrationTransform,
    kernel,
    raw_kernel,
    *,
    samples_per_condition: int,
    valid_conditions: list[int],
    generator: torch.Generator,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float]]:
    samples_by_condition: dict[str, Any] = {}
    val_mmds = []
    raw_val_mmds = []
    sample_kls = []

    policy.model.eval()
    base.model.eval()
    for condition_idx in tqdm(valid_conditions, desc="Final validation sampling"):
        key = condition_key(condition_idx)
        cond = sample_condition_matrix(
            valid_cache,
            condition_idx,
            samples_per_condition,
            generator,
        )
        sample = policy.sample(cond)
        tokens = sample.tokens
        log_p_finetuned = sample.log_p_theta.detach().cpu()
        log_p_base = base.log_p_total(sample).detach().cpu()
        condition_sample_kl = (log_p_finetuned - log_p_base).mean()
        sample_kls.append(condition_sample_kl)

        raw_features = feature_extractor(tokens).detach().cpu().to(torch.float32)
        log_features, calibration_features = transform_generated_features(
            raw_features,
            feature_transform,
            device,
        )

        target_idx = valid_cache.condition_to_indices[key]
        target_features = valid_cache.calibration_features[target_idx].to(device)
        val_mmd = estimate_condition_mmd(calibration_features, target_features, kernel)
        raw_val_mmd = estimate_condition_mmd(
            calibration_features,
            target_features,
            raw_kernel,
        )
        if torch.isfinite(val_mmd):
            val_mmds.append(val_mmd.detach().cpu())
        if torch.isfinite(raw_val_mmd):
            raw_val_mmds.append(raw_val_mmd.detach().cpu())

        samples_by_condition[key] = {
            "sequence_tokens": tokens.cpu(),
            "condition_matrix": cond.cpu(),
            "log_p_finetuned": log_p_finetuned,
            "log_p_base": log_p_base,
            "sample_kl": float(condition_sample_kl),
            "raw_alphagenome_features": raw_features.cpu(),
            "log1p_alphagenome_features": log_features.detach().cpu(),
            "calibration_alphagenome_features": calibration_features.detach().cpu(),
            "val_mmd": float(val_mmd.detach().cpu()),
            "raw_val_mmd": float(raw_val_mmd.detach().cpu()),
        }

    metrics = {}
    if val_mmds:
        metrics["final/mmd"] = torch.stack(val_mmds).mean().item()
    else:
        metrics["final/mmd"] = float("nan")
    if raw_val_mmds:
        metrics["final/raw_mmd"] = torch.stack(raw_val_mmds).mean().item()
    else:
        metrics["final/raw_mmd"] = float("nan")
    metrics["final/sample_kl"] = torch.stack(sample_kls).mean().item()
    return samples_by_condition, metrics


def auto_scale_conditional_kernel(
    policy: TauLeapingDiffusionPolicy,
    train_cache: SplitCache,
    feature_transform: CalibrationTransform,
    feature_extractor,
    kernel,
    condition_indices: list[int],
    *,
    batch_size: int,
    use_loo: bool,
    device: torch.device,
) -> tuple[Any, float, float, list[int]]:
    scaling_generator = torch.Generator().manual_seed(AUTO_SCALE_SEED)
    condition_order = torch.randperm(
        len(condition_indices), generator=scaling_generator
    )
    scale_condition_indices = [
        condition_indices[int(idx)]
        for idx in condition_order[: min(AUTO_SCALE_CONDITIONS, len(condition_indices))]
    ]

    estimates = []
    policy.model.eval()
    fork_devices = []
    if device.type == "cuda":
        fork_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(AUTO_SCALE_SEED)
        for condition_idx in tqdm(
            scale_condition_indices,
            desc="Sampling for kernel auto-scaling",
        ):
            key = condition_key(condition_idx)
            cond = sample_condition_matrix(
                train_cache,
                condition_idx,
                batch_size,
                scaling_generator,
            )
            sample = policy.sample(cond)
            raw_features = feature_extractor(sample.tokens).detach()
            _, generated_features = transform_generated_features(
                raw_features,
                feature_transform,
                device,
            )

            target_idx = train_cache.condition_to_indices[key]
            target_features = train_cache.calibration_features[target_idx].to(device)
            kyy_mean = _mean_off_diagonal_kernel(target_features, kernel)
            estimate = _mmd_calibration_terms(
                generated_features,
                target_features,
                kernel,
                kyy_mean=kyy_mean,
                use_loo=use_loo,
                self_repulsion_weight=1.0,
            ).estimate
            estimates.append(estimate.detach())

    initial_constraint = float(torch.stack(estimates).mean().item())
    eps = torch.finfo(torch.float32).eps
    if initial_constraint <= eps:
        raise ValueError(
            "Initial constraint estimate must be positive to auto-scale the "
            f"kernel, got {initial_constraint:.6g}."
        )
    scale_factor = AUTO_SCALE_TARGET_CONSTRAINT / initial_constraint
    return (
        scale_kernel(kernel, scale_factor),
        scale_factor,
        initial_constraint,
        scale_condition_indices,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("MMD coefficients require --batch-size >= 2.")
    if not args.no_loo and args.batch_size < 3:
        raise ValueError("MMD LOO coefficients require --batch-size >= 3.")
    if args.final_samples_per_condition < 2:
        raise ValueError("--final-samples-per-condition must be at least 2.")
    if not (0.0 < args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in (0, 1].")

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.target_cache_pt, map_location="cpu", weights_only=True)
    train_raw = payload["splits"]["train"]["features"].to(torch.float32)
    train_log_features = torch.log1p(train_raw)
    feature_transform_cpu = fit_feature_transform(args, train_log_features)
    feature_transform_device = to_device_transform(feature_transform_cpu, device)
    train_cache = load_split_cache(payload, "train", feature_transform_cpu)
    valid_cache = load_split_cache(payload, "valid", feature_transform_cpu)

    train_conditions = valid_condition_indices(train_cache, min_count=2)
    valid_conditions = valid_condition_indices(valid_cache, min_count=2)
    condition_indices = sorted(set(train_conditions).intersection(valid_conditions))
    if not condition_indices:
        raise ValueError(
            "No conditions have at least two train and validation targets."
        )

    policy_model = load_model(args.checkpoint_path, device)
    base_model = copy.deepcopy(policy_model)
    freeze_model(base_model)

    seq_len = int(payload["splits"]["train"]["sequence_length"])
    policy = TauLeapingDiffusionPolicy(
        policy_model,
        seq_len=seq_len,
        sample_steps=args.sample_steps,
        temperature=args.temperature,
        autocast_dtype=args.diffusion_autocast_dtype,
        device=device,
    )
    base = TauLeapingDiffusionPolicy(
        base_model,
        seq_len=seq_len,
        sample_steps=args.sample_steps,
        temperature=args.temperature,
        autocast_dtype=args.diffusion_autocast_dtype,
        device=device,
    )

    feature_extractor = make_feature_extractor(args)
    match args.kernel:
        case "energy":
            raw_kernel = energy_distance_kernel()
        case "dotproduct":
            raw_kernel = dot_product_kernel
        case _:
            raise ValueError(f"Unknown kernel {args.kernel}")
    kernel = raw_kernel
    (
        kernel,
        kernel_scale_factor,
        kernel_scale_initial_constraint,
        kernel_scale_condition_indices,
    ) = auto_scale_conditional_kernel(
        policy,
        train_cache,
        feature_transform_device,
        feature_extractor,
        kernel,
        condition_indices,
        batch_size=args.batch_size,
        use_loo=not args.no_loo,
        device=device,
    )
    args.kernel_scale = kernel_scale_factor
    args.kernel_scale_initial_constraint = kernel_scale_initial_constraint
    args.kernel_scale_target_constraint = AUTO_SCALE_TARGET_CONSTRAINT
    args.kernel_scale_conditions = kernel_scale_condition_indices
    args.kernel_scale_seed = AUTO_SCALE_SEED

    train_kyy_mean = {
        condition_key(condition_idx): _mean_off_diagonal_kernel(
            train_cache.calibration_features[
                train_cache.condition_to_indices[condition_key(condition_idx)]
            ].to(device),
            kernel,
        ).detach()
        for condition_idx in condition_indices
    }

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "cosine":
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            total_epochs=args.epochs,
            warmup_epochs=0,
            min_lr_ratio=args.min_lr_ratio,
        )
    else:
        scheduler = None

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            config=vars(args),
        )

    print(
        f"Calibrating {len(condition_indices)} conditions, seq_len={seq_len}, "
        f"sample_steps={policy.num_sample_chunks}, batch_size={args.batch_size}, "
        f"kernel={args.kernel}, kernel_scale={kernel_scale_factor:.6g}, "
        f"kernel_scale_initial_constraint={kernel_scale_initial_constraint:.6g}, "
        f"feature_transform={args.feature_transform}, mmd_loo={not args.no_loo}, "
        f"lr_schedule={args.lr_schedule}"
    )

    global_step = 0
    for epoch in range(args.epochs):
        # Keep dropout disabled so sampled trajectories and replayed log-probs
        # are evaluated under the same policy.
        policy_model.eval()
        epoch_order = torch.randperm(len(condition_indices), generator=generator)
        epoch_step_metrics: list[dict[str, float]] = []
        pbar = tqdm(epoch_order.tolist(), desc=f"Epoch {epoch}")
        for condition_order_idx in pbar:
            condition_idx = int(condition_indices[condition_order_idx])
            key = condition_key(condition_idx)
            cond = sample_condition_matrix(
                train_cache,
                condition_idx,
                args.batch_size,
                generator,
            )

            with torch.no_grad():
                sample = policy.sample(cond)
                raw_features = feature_extractor(sample.tokens).detach()
                _, generated_features = transform_generated_features(
                    raw_features,
                    feature_transform_device,
                    device,
                )

                target_idx = train_cache.condition_to_indices[key]
                target_features = train_cache.calibration_features[target_idx].to(
                    device
                )
                mmd_terms = _mmd_calibration_terms(
                    generated_features,
                    target_features,
                    kernel,
                    kyy_mean=train_kyy_mean[key],
                    use_loo=not args.no_loo,
                    self_repulsion_weight=1.0,
                )
                c_mmd = mmd_terms.coeff
                mmd_estimate = mmd_terms.estimate

                log_p_theta = sample.log_p_theta.to(device)
                log_p_base = base.log_p_total(sample).to(device)
                c_kl, kl_loss_val = compute_kl_coefficients(log_p_theta, log_p_base)

                c_total, total_loss_val, kl_weight, constraint_weight = (
                    combine_coefficients(
                        c_kl,
                        c_mmd,
                        kl_loss_val,
                        mmd_estimate,
                        args.lambd,
                        args.loss_weighting,
                    )
                )
                c_total = c_total.detach()

            optimizer.zero_grad(set_to_none=True)
            for sample_idx in range(policy.num_sample_chunks):
                delta = policy.log_p(sample, sample_idx=sample_idx)
                loss_piece = (c_total * delta).sum()
                loss_piece.backward()

            grad_norm_before = grad_norm(policy_model.parameters())
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    policy_model.parameters(), args.grad_clip
                )
            grad_norm_after = grad_norm(policy_model.parameters())
            optimizer.step()

            metrics = {
                "step/epoch": epoch,
                "step/condition_idx": condition_idx,
                "step/loss": total_loss_val.item(),
                "step/mmd": mmd_estimate.item(),
                "step/kl": kl_loss_val.item(),
                "step/lr": optimizer.param_groups[0]["lr"],
                "step/kl_weight": kl_weight,
                "step/constraint_weight": constraint_weight,
                "step/grad_norm": grad_norm_before,
                "step/grad_norm_clipped": grad_norm_after,
                **prefixed_abs_stats("step", "c_mmd", c_mmd),
                **prefixed_abs_stats("step", "c_kl", c_kl),
                **prefixed_abs_stats("step", "c_total", c_total),
            }
            epoch_step_metrics.append(metrics)
            pbar.set_postfix(
                loss=f"{metrics['step/loss']:.4f}",
                mmd=f"{metrics['step/mmd']:.4f}",
                kl=f"{metrics['step/kl']:.4f}",
                condition=condition_idx,
            )
            if wandb_run is not None and global_step % args.log_every == 0:
                wandb_run.log(metrics, step=global_step)
            global_step += 1
        epoch_metrics = mean_metrics(epoch_step_metrics, "step")
        epoch_metrics["epoch/index"] = epoch
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=global_step)
        if scheduler is not None:
            scheduler.step()

    final_samples, final_metrics = final_samples_and_metrics(
        policy,
        base,
        valid_cache,
        feature_extractor,
        feature_transform_device,
        kernel,
        raw_kernel,
        samples_per_condition=args.final_samples_per_condition,
        valid_conditions=condition_indices,
        generator=generator,
        device=device,
    )
    if wandb_run is not None:
        wandb_run.log(final_metrics, step=global_step)

    feature_transform_payload = serialize_feature_transform(feature_transform_cpu, args)
    torch.save(
        {
            "state_dict": policy_model.state_dict(),
            "source_checkpoint_path": str(args.checkpoint_path),
            "args": vars(args),
            "feature_transform": feature_transform_payload,
            "pca": feature_transform_cpu if args.feature_transform == "pca" else None,
        },
        args.output_dir / "calibrated_model.pt",
    )
    torch.save(
        {
            "samples_by_condition": final_samples,
            "metrics": final_metrics,
            "condition_indices": condition_indices,
            "feature_transform": feature_transform_payload,
            "pca": feature_transform_cpu if args.feature_transform == "pca" else None,
            "args": vars(args),
        },
        args.output_dir / "final_samples.pt",
    )
    print(f"Saved calibrated model to {args.output_dir / 'calibrated_model.pt'}")
    print(f"Saved final samples to {args.output_dir / 'final_samples.pt'}")
    print(f"Final validation MMD mean: {final_metrics['final/mmd']:.6f}")
    print(f"Final raw validation MMD mean: {final_metrics['final/raw_mmd']:.6f}")
    print(f"Final sample KL mean: {final_metrics['final/sample_kl']:.6f}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
