import argparse
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

import secstruct
from cgm.cgm_distribution import (
    calibrate_mmd,
    dot_product_kernel,
    energy_distance_kernel,
    median_heuristic_rbf_sigmas,
    rbf_mixture_kernel,
)
from cgm.cgm import log_p_chunked
from cgm.utils import CheckpointEveryN, clone_network, default_logger
from genie_score_network import GenieScoreNetwork, diffusion, drift

sys.path.append("../neural_sde")
from neural_sde import NeuralSDE
from utils import BackwardDrift


def clean_df(csv_path: str) -> pd.DataFrame:
    """
    Read CATH secondary-structure CSV and keep valid alpha/beta rows.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"% Helix": "alpha_helix", "% Strand": "beta_strand"})
    df = (
        df[["alpha_helix", "beta_strand"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    df = df[df["alpha_helix"].between(0, 1) & df["beta_strand"].between(0, 1)]
    df = df[df["alpha_helix"] + df["beta_strand"] <= 1.0]
    if df.empty:
        raise ValueError("No valid CATH rows remain after cleaning.")
    return df


def build_hstar(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Build [m, 2] target samples [alpha_helix, beta_strand].
    """
    return torch.tensor(
        df[["alpha_helix", "beta_strand"]].to_numpy(),
        device=device,
        dtype=torch.float32,
    )


def compute_univariate_thresholds(
    df: pd.DataFrame, n_quantiles: int
) -> dict[str, list[float]]:
    quantiles = torch.linspace(0.0, 1.0, n_quantiles + 2)[1:-1].tolist()
    alpha_q = df["alpha_helix"].quantile(quantiles, interpolation="linear")
    beta_q = df["beta_strand"].quantile(quantiles, interpolation="linear")
    return {"alpha": alpha_q.tolist(), "beta": beta_q.tolist()}


def compute_bivariate_thresholds(n_quantiles: int) -> dict[str, list[float]]:
    helix_ths = torch.linspace(0.0, 1.0, n_quantiles + 2)[1:].tolist()
    strand_ths = torch.linspace(0.0, 1.0, n_quantiles + 2)[1:].tolist()
    alpha_thresholds: list[float] = []
    beta_thresholds: list[float] = []
    for helix_thresh in helix_ths:
        for strand_thresh in strand_ths:
            if helix_thresh == 1.0 and strand_thresh == 1.0:
                continue
            alpha_thresholds.append(helix_thresh)
            beta_thresholds.append(strand_thresh)
    return {"alpha": alpha_thresholds, "beta": beta_thresholds}


def univariate_cdf_features(
    fractions: torch.Tensor,
    thresholds: dict[str, list[float]],
) -> torch.Tensor:
    alpha_thresholds = torch.tensor(
        thresholds["alpha"], device=fractions.device, dtype=fractions.dtype
    )
    beta_thresholds = torch.tensor(
        thresholds["beta"], device=fractions.device, dtype=fractions.dtype
    )
    alpha_features = (fractions[:, [0]] <= alpha_thresholds).to(fractions.dtype)
    beta_features = (fractions[:, [1]] <= beta_thresholds).to(fractions.dtype)
    return torch.cat([alpha_features, beta_features], dim=1)


def bivariate_cdf_features(
    fractions: torch.Tensor,
    thresholds: dict[str, list[float]],
) -> torch.Tensor:
    alpha_thresholds = torch.tensor(
        thresholds["alpha"], device=fractions.device, dtype=fractions.dtype
    )
    beta_thresholds = torch.tensor(
        thresholds["beta"], device=fractions.device, dtype=fractions.dtype
    )
    return (
        (fractions[:, [0]] <= alpha_thresholds) & (fractions[:, [1]] <= beta_thresholds)
    ).to(fractions.dtype)


def make_kernel(
    kernel_name: str,
    hstar: torch.Tensor,
    beta: float,
) -> tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor | None]:
    if kernel_name == "energy":
        return energy_distance_kernel(beta=beta), None
    if kernel_name == "rbf":
        sigmas = median_heuristic_rbf_sigmas(hstar)
        return rbf_mixture_kernel(sigmas=sigmas), sigmas
    if kernel_name == "dot":
        return dot_product_kernel, None
    raise ValueError(f"Unknown kernel {kernel_name}")


def parse_kernel_scale(value: str) -> float | str:
    if value == "auto":
        return value
    scale = float(value)
    if scale <= 0:
        raise argparse.ArgumentTypeError(
            f"kernel_scale must be positive or 'auto', got {value!r}"
        )
    return scale


def h_alpha_beta(x: Any) -> torch.Tensor:
    """
    Extract raw secondary-structure features [alpha, beta] from final coordinates.
    """
    x_last = x.xs[:, -1]
    h_vals = []
    for b in range(x_last.shape[0]):
        x_b = x_last[b].reshape(-1, 3).detach().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            pdb_path = tmp.name
        secstruct.save_pdb(x_b, pdb_path)
        try:
            ss_fracs = secstruct.sec_struct_frac(pdb_path)
        finally:
            os.remove(pdb_path)
        h_vals.append(
            torch.tensor(
                [ss_fracs[0], ss_fracs[1]],
                device=x_last.device,
                dtype=torch.float32,
            )
        )
    return torch.stack(h_vals, dim=0)


def make_feature_extractor(
    feature_fn: Callable[[torch.Tensor, dict[str, list[float]]], torch.Tensor],
    thresholds: dict[str, list[float]],
) -> Callable[[Any], torch.Tensor]:
    def extractor(x: Any) -> torch.Tensor:
        return feature_fn(h_alpha_beta(x), thresholds)

    return extractor


def build_feature_map(
    feature_map: str,
    df: pd.DataFrame,
    device: torch.device,
    n_quantiles: int,
) -> tuple[torch.Tensor, Callable[[Any], torch.Tensor], dict[str, Any]]:
    raw_target_samples = build_hstar(df, device=device)

    if feature_map == "raw":
        return raw_target_samples, h_alpha_beta, {"feature_map": feature_map}

    if n_quantiles < 1:
        raise ValueError("n_quantiles must be at least 1 when using CDF features.")

    if feature_map == "cdf_uni":
        thresholds = compute_univariate_thresholds(df, n_quantiles)
        return (
            univariate_cdf_features(raw_target_samples, thresholds),
            make_feature_extractor(univariate_cdf_features, thresholds),
            {
                "feature_map": feature_map,
                "n_quantiles": n_quantiles,
                "cdf_thresholds": thresholds,
            },
        )

    if feature_map == "cdf_bi":
        thresholds = compute_bivariate_thresholds(n_quantiles)
        return (
            bivariate_cdf_features(raw_target_samples, thresholds),
            make_feature_extractor(bivariate_cdf_features, thresholds),
            {
                "feature_map": feature_map,
                "n_quantiles": n_quantiles,
                "cdf_thresholds": thresholds,
            },
        )

    raise ValueError(f"Unknown feature map {feature_map}")


def training_logger(
    logs: dict[str, Any],
    model: NeuralSDE,
    base_model: NeuralSDE,
    x: Any,
    batch_chunks: int,
    samp_chunks: int,
    use_wandb: bool,
) -> None:
    """
    Log training metrics either to wandb or stdout.
    """
    metric_logs = {
        "epoch": logs["epoch"],
        "kl_loss": logs["kl_loss"],
        "moment_loss": logs["constraint_loss"],
        "loss": logs["loss"],
    }

    if use_wandb:
        wandb.log(metric_logs)
    else:
        default_logger(metric_logs)


def make_t_grid(device: torch.device) -> torch.Tensor:
    """
    Time grid for discretizing diffusion sampling.
    """
    n_steps = 100
    mid_pt = 0.05
    t_grid_1 = torch.linspace(0.0, mid_pt, n_steps // 2, device=device)
    t_grid_2 = torch.linspace(
        mid_pt + t_grid_1[-1] - t_grid_1[-2],
        1.0,
        (n_steps + 1) - n_steps // 2,
        device=device,
    )
    return torch.concat([t_grid_1, t_grid_2])


def build_base_model(
    device: torch.device,
    noise_scale: float,
    L: int,
    max_sample_batch: int | None = None,
) -> NeuralSDE:
    """
    Construct the base Genie2 neural-SDE.
    """
    t_grid = make_t_grid(device)
    genie_score = GenieScoreNetwork(device, noise_scale, None)
    sqrt_alphas = genie_score.denoiser.sqrt_alphas
    sqrt_betas = genie_score.denoiser.sqrt_betas

    f = lambda x, t: drift(x, t, sqrt_alphas)
    sigma = lambda t: diffusion(t, sqrt_betas, noise_scale)
    genie_drift_network = BackwardDrift(genie_score, f, sigma)
    return NeuralSDE(
        sde_dim=L * 3,
        drift=genie_drift_network,
        diffusion=lambda t: sigma(1.0 - t),
        t_grid=t_grid,
        max_sample_batch=max_sample_batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--kernel", choices=["energy", "rbf", "dot"], default="energy")
    parser.add_argument(
        "--loss_weighting",
        choices=["raw", "normalized"],
        default="raw",
        help="Whether lambd sets a raw KL weight or a normalized KL:constraint ratio.",
    )
    parser.add_argument(
        "--kernel_scale",
        type=parse_kernel_scale,
        default=None,
        help="Optional kernel scaling factor, or 'auto' to autoscale from initial batches.",
    )
    parser.add_argument(
        "--feature_map",
        choices=["raw", "cdf_uni", "cdf_bi"],
        default="raw",
    )
    parser.add_argument("--N_quantiles", type=int, default=9)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Energy-distance exponent beta. Only used when --kernel=energy.",
    )
    parser.add_argument("--self_repulsion_weight", type=float, default=1.0)
    parser.add_argument("--root_dir", type=str, default=None)
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_init", type=float, default=1e-5)
    parser.add_argument("--lr_min", type=float, default=1e-7)
    parser.add_argument("--samp_chunks", type=int, default=20)
    parser.add_argument("--batch_chunks", type=int, default=4)
    parser.add_argument("--max_sample_batch", type=int, default=None)
    parser.add_argument("--noise_scale", type=float, default=0.5)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--N_samples", type=int, default=10**3)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_run_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_name = (
        f"{args.feature_map}_{args.kernel}_lambd_{str(args.lambd).replace('.', '_')}"
    )
    if args.loss_weighting != "raw":
        run_name += f"_losswt_{args.loss_weighting}"
    if args.kernel_scale is not None:
        kernel_scale_tag = str(args.kernel_scale).replace(".", "_")
        run_name += f"_ks_{kernel_scale_tag}"
    root_dir = (
        Path(args.root_dir)
        if args.root_dir is not None
        else Path("genie_outputs") / run_name
    )
    pdb_dir = root_dir / "pdbs"
    return root_dir, pdb_dir


def make_checkpoint_fn(
    args: argparse.Namespace,
    root_dir: Path,
) -> CheckpointEveryN | None:
    if args.epochs > 0:
        ckpt_dir = root_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return CheckpointEveryN(ckpt_dir=str(ckpt_dir), N=args.epochs)
    return None


def save_sample_outputs(
    model: NeuralSDE,
    base_model: NeuralSDE,
    n_samples: int,
    root_dir: Path,
    pdb_dir: Path,
    sample_batch_size: int,
    batch_chunks: int,
    samp_chunks: int,
) -> None:
    if n_samples <= 0:
        return

    pdb_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | float | str]] = []

    sample_idx = 0
    with torch.no_grad(), tqdm(total=n_samples, desc="Sampling") as pbar:
        while sample_idx < n_samples:
            curr_batch_size = min(sample_batch_size, n_samples - sample_idx)
            batch = model.sample(curr_batch_size)
            batch_samples = batch.xs[:, -1, :].detach().cpu().numpy()
            curr_batch_chunks = min(batch_chunks, curr_batch_size)
            log_p_model = log_p_chunked(
                model,
                batch,
                curr_batch_size,
                curr_batch_chunks,
                samp_chunks,
            ).detach()
            log_p_base = log_p_chunked(
                base_model,
                batch,
                curr_batch_size,
                curr_batch_chunks,
                samp_chunks,
            ).detach()

            for batch_sample, lp_model, lp_base in zip(
                batch_samples,
                log_p_model.cpu().tolist(),
                log_p_base.cpu().tolist(),
            ):
                x0 = batch_sample.reshape(batch_sample.shape[0] // 3, 3)
                pdb_path = pdb_dir / f"sample_{sample_idx}.pdb"
                secstruct.save_pdb(x0, str(pdb_path))
                ss_fracs = secstruct.sec_struct_frac(str(pdb_path))
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "pdb_path": str(pdb_path),
                        "alpha_helix": float(ss_fracs[0]),
                        "beta_strand": float(ss_fracs[1]),
                        "log_p_model": lp_model,
                        "log_p_base": lp_base,
                        "kl_sample": lp_model - lp_base,
                    }
                )
                sample_idx += 1
                pbar.update(1)

    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "sample_secstruct.csv", index=False)
    print(f"Estimated KL(model || base) = {df['kl_sample'].mean():.4f} nats")


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    root_dir, pdb_dir = build_run_paths(args)
    ckpt_fn = make_checkpoint_fn(args, root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = clean_df(args.path)
    hstar, h_fn, feature_config = build_feature_map(
        args.feature_map,
        df,
        device=device,
        n_quantiles=args.N_quantiles,
    )
    kernel, rbf_sigmas = make_kernel(args.kernel, hstar, args.beta)
    if rbf_sigmas is not None:
        print(f"Using RBF kernel with sigmas: {rbf_sigmas.detach().cpu().tolist()}")
    model = build_base_model(
        device=device,
        noise_scale=args.noise_scale,
        L=args.L,
        max_sample_batch=args.max_sample_batch,
    )
    eval_base_model = clone_network(model)

    run = None
    if args.wandb_project is not None:
        run = wandb.init(
            project=args.wandb_project,
            config={
                "noise_scale": args.noise_scale,
                "num_residues": args.L,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "lr_init": args.lr_init,
                "lr_min": args.lr_min,
                "lambd": args.lambd,
                "loss_weighting": args.loss_weighting,
                "kernel_scale": args.kernel_scale,
                "root_dir": str(root_dir),
                "kernel": args.kernel,
                "feature_map": args.feature_map,
                "N_quantiles": args.N_quantiles,
                "beta": args.beta,
                "self_repulsion_weight": args.self_repulsion_weight,
                "rbf_sigmas": (
                    None if rbf_sigmas is None else rbf_sigmas.detach().cpu().tolist()
                ),
                **feature_config,
            },
        )

    calibrated_model = calibrate_mmd(
        model=model,
        h=h_fn,
        hstar=hstar,
        lambd=args.lambd,
        loss_weighting=args.loss_weighting,
        kernel=kernel,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_params={"lr": args.lr_init},
        lr_scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params={"T_max": args.epochs, "eta_min": args.lr_min},
        samp_chunks=args.samp_chunks,
        batch_chunks=args.batch_chunks,
        use_loo=True,
        self_repulsion_weight=args.self_repulsion_weight,
        kernel_scale=args.kernel_scale,
        logger=partial(
            training_logger,
            batch_chunks=args.batch_chunks,
            samp_chunks=args.samp_chunks,
            use_wandb=run is not None,
        ),
        checkpoint_fn=ckpt_fn,
        disable_pbar=False,
    )

    if args.N_samples > 0:
        save_sample_outputs(
            model=calibrated_model,
            base_model=eval_base_model,
            n_samples=args.N_samples,
            root_dir=root_dir,
            pdb_dir=pdb_dir,
            sample_batch_size=args.max_sample_batch or args.batch_size,
            batch_chunks=args.batch_chunks,
            samp_chunks=args.samp_chunks,
        )

    if run is not None:
        run.finish()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
