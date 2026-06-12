"""
Calibrate a pretrained G2PT model toward a smiley-face distribution in the
standardized (MolWt, MolLogP) plane.

The target distribution is synthetic:
  - two Gaussian eyes
  - one noisy lower arc for the smile

MolWt and MolLogP are standardized using samples drawn from the pretrained
base model so the smiley is expressed relative to the model's native descriptor
scale. Invalid generated molecules are mapped to (0, 0), so the target
distribution leaves a hole around the origin.

Usage:
    PYTHONPATH=.. /Users/ndiamant/miniforge3/envs/g2pt/bin/python \
        calibrate_g2pt_smiley.py \
        --out_root /path/to/output_root \
        --lambd 0.1
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cgm import utils
from cgm.cgm_distribution import (
    calibrate_mmd,
    energy_distance_kernel,
    rbf_mixture_kernel,
    scale_kernel,
)
from datasets_utils import get_smiles, seq_to_mol, seq_to_molecule_with_partial_charges
from features import rdkit_descriptor_df
from g2pt_cgm_model import G2PTModel


def sample_smiley_face(
    n: int,
    *,
    eye_frac: float = 0.12,
    eye_sigma: float = 0.16,
    mouth_noise: float = 0.12,
    hole_radius: float = 0.35,
    seed: int = 0,
) -> torch.Tensor:
    """
    Sample points from a simple smiley-face distribution in standardized 2D
    coordinates.
    """
    rng = np.random.default_rng(seed)

    n_eye_left = int(n * eye_frac)
    n_eye_right = int(n * eye_frac)
    n_mouth = n - n_eye_left - n_eye_right

    left_eye_center = np.array([-0.8, 0.75], dtype=np.float32)
    right_eye_center = np.array([0.8, 0.75], dtype=np.float32)

    left_eye = left_eye_center + eye_sigma * rng.standard_normal((n_eye_left, 2))
    right_eye = right_eye_center + eye_sigma * rng.standard_normal((n_eye_right, 2))

    theta = rng.uniform(np.deg2rad(210.0), np.deg2rad(330.0), size=n_mouth)
    mouth_center = np.array([0.0, 0.15], dtype=np.float32)
    mouth_radius = 1.0
    mouth = np.column_stack(
        [
            mouth_center[0] + mouth_radius * np.cos(theta),
            mouth_center[1] + mouth_radius * np.sin(theta),
        ]
    )
    mouth += mouth_noise * rng.standard_normal((n_mouth, 2))

    x = np.vstack([left_eye, right_eye, mouth]).astype(np.float32)
    r = np.linalg.norm(x, axis=1)
    keep = r >= hole_radius
    if keep.all():
        return torch.from_numpy(x)

    kept = x[keep]
    refill = sample_smiley_face(
        n - kept.shape[0],
        eye_frac=eye_frac,
        eye_sigma=eye_sigma,
        mouth_noise=mouth_noise,
        hole_radius=hole_radius,
        seed=int(rng.integers(1_000_000_000)),
    )
    return torch.cat([torch.from_numpy(kept), refill], dim=0)


def is_guacamol_model(model_name: str) -> bool:
    return "guacamol" in model_name.lower()


def sequence_to_smiles(seq_str: str, model_name: str) -> str | None:
    try:
        if is_guacamol_model(model_name):
            mol = seq_to_molecule_with_partial_charges(seq_str)
        else:
            mol = seq_to_mol(seq_str)
        return get_smiles(mol)
    except Exception:
        return None


def decode_token_ids_to_smiles(
    token_ids: torch.Tensor,
    tokenizer,
    model_name: str,
    max_workers: int = 4,
) -> list[str | None]:
    seq_strs = tokenizer.batch_decode(token_ids)
    if max_workers == 1:
        return [sequence_to_smiles(seq_str, model_name) for seq_str in seq_strs]

    return thread_map(
        partial(sequence_to_smiles, model_name=model_name),
        seq_strs,
        max_workers=max_workers,
        disable=True,
        chunksize=512,
    )


def descriptor_panel(
    smiles_list: list[str | None],
) -> tuple[pd.DataFrame, torch.Tensor]:
    """
    Return RDKit descriptor dataframe and a dense [N, 2] tensor with raw
    (MolWt, MolLogP) values. Invalid molecules get zeros in the tensor and
    valid=False in the dataframe.
    """
    df = rdkit_descriptor_df(smiles_list)
    out = torch.zeros(len(df), 2, dtype=torch.float32)
    valid_mask_np = df["valid"].to_numpy(dtype=bool)
    if valid_mask_np.any():
        out[torch.from_numpy(valid_mask_np)] = torch.tensor(
            df.loc[valid_mask_np, ["MolWt", "MolLogP"]].to_numpy(),
            dtype=torch.float32,
        )
    return df, out


def fit_standardization(
    smiles_list: list[str | None],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fit mean and standard deviation for (MolWt, MolLogP) on valid molecules.
    """
    df, raw = descriptor_panel(smiles_list)
    valid_mask = torch.from_numpy(df["valid"].to_numpy(dtype=bool))
    if not valid_mask.any():
        raise ValueError("No valid molecules available to fit descriptor scaling.")

    valid_raw = raw[valid_mask]
    mu = valid_raw.mean(dim=0)
    sigma = valid_raw.std(dim=0).clamp_min(1e-6)
    return mu, sigma, int(valid_mask.sum().item())


def standardize_descriptors(
    raw: torch.Tensor,
    df: pd.DataFrame,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Standardize raw (MolWt, MolLogP) values. Invalid molecules remain at (0, 0).
    """
    z = torch.zeros_like(raw)
    valid_mask = torch.from_numpy(df["valid"].to_numpy(dtype=bool))
    print(f"valid frac: {df['valid'].mean():.1%}")
    if valid_mask.any():
        z[valid_mask] = (raw[valid_mask] - mu) / sigma
    return z


def build_standardized_descriptor_table(
    smiles_list: list[str | None],
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> pd.DataFrame:
    df, raw = descriptor_panel(smiles_list)
    z = standardize_descriptors(raw, df, mu, sigma)
    table = df.loc[:, ["smiles", "valid", "MolWt", "MolLogP"]].copy()
    table["MolWt_z"] = z[:, 0].numpy()
    table["MolLogP_z"] = z[:, 1].numpy()
    return table


def presample_base_model_for_standardization(
    model: G2PTModel,
    model_name: str,
    n_samples: int,
    batch_size: int,
    out_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, int, Path, torch.Tensor]:
    was_training = model.training
    model.eval()
    smiles_list: list[str | None] = []

    try:
        pbar = tqdm(total=n_samples, desc="Presampling base model")
        n_done = 0
        while n_done < n_samples:
            n_batch = min(batch_size, n_samples - n_done)
            with torch.no_grad():
                samples = model.sample(n_batch)
            smiles_list.extend(
                decode_token_ids_to_smiles(
                    samples.token_ids,
                    model.tokenizer,
                    model_name,
                )
            )
            n_done += n_batch
            pbar.update(n_batch)
        pbar.close()
    finally:
        if was_training:
            model.train()

    mu, sigma, n_valid = fit_standardization(smiles_list)
    table = build_standardized_descriptor_table(smiles_list, mu, sigma)
    csv_path = out_dir / "presampled_norm_samples.csv"
    table.to_csv(csv_path, index=False)
    presample_z = torch.from_numpy(
        table.loc[:, ["MolWt_z", "MolLogP_z"]].to_numpy(dtype=np.float32)
    )

    print(f"Saved {n_samples} presampled normalization rows to {csv_path}")
    print(
        "Descriptor standardization fit on "
        f"{n_valid}/{n_samples} valid base-model samples."
    )
    print(f"  mean = {mu.tolist()}")
    print(f"  std = {sigma.tolist()}")
    return mu, sigma, n_valid, csv_path, presample_z


def make_h(
    tokenizer,
    model_name: str,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> Callable:
    def h(x) -> torch.Tensor:
        smiles = decode_token_ids_to_smiles(x.token_ids, tokenizer, model_name)
        df, raw = descriptor_panel(smiles)
        return standardize_descriptors(raw, df, mu, sigma)

    return h


def save_training_sample_snapshot(
    model: G2PTModel,
    model_name: str,
    n_samples: int,
    batch_size: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    out_dir: Path,
    epoch: int,
    sample_seed: int,
) -> Path:
    was_training = model.training
    model.eval()
    smiles_list: list[str | None] = []

    try:
        with torch.random.fork_rng():
            torch.manual_seed(sample_seed)
            n_done = 0
            while n_done < n_samples:
                n_batch = min(batch_size, n_samples - n_done)
                with torch.no_grad():
                    samples = model.sample(n_batch)
                smiles_list.extend(
                    decode_token_ids_to_smiles(
                        samples.token_ids,
                        model.tokenizer,
                        model_name,
                    )
                )
                n_done += n_batch
    finally:
        if was_training:
            model.train()

    table = build_standardized_descriptor_table(smiles_list, mu, sigma)
    table.insert(0, "sample_index", range(len(table)))

    snapshot_dir = out_dir / "training_samples"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = snapshot_dir / f"epoch_{epoch + 1:05d}.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved {n_samples} training samples to {csv_path}")
    return csv_path


def make_logger(
    tokenizer,
    model_name: str,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    dict_logger: utils.DictLogger,
    out_dir: Path,
    sample_every_epochs: int,
    n_training_samples: int,
    training_sample_seed: int,
    batch_size: int,
) -> Callable:
    def logger(logs: dict, model: G2PTModel, base_model: G2PTModel, xs) -> None:
        smiles = decode_token_ids_to_smiles(xs.token_ids, tokenizer, model_name)
        df, raw = descriptor_panel(smiles)
        z = standardize_descriptors(raw, df, mu, sigma)
        scalars = {k: v for k, v in logs.items() if k != "h_bar"}
        scalars["validity"] = float(df["valid"].mean())
        scalars["molwt_mean_z"] = z[:, 0].mean().item()
        scalars["mollogp_mean_z"] = z[:, 1].mean().item()
        dict_logger(scalars)
        utils.default_logger(scalars)

        epoch = int(logs["epoch"])
        if sample_every_epochs > 0 and (epoch + 1) % sample_every_epochs == 0:
            save_training_sample_snapshot(
                model=model,
                model_name=model_name,
                n_samples=n_training_samples,
                batch_size=batch_size,
                mu=mu,
                sigma=sigma,
                out_dir=out_dir,
                epoch=epoch,
                sample_seed=training_sample_seed,
            )

    return logger


def save_results(metrics: dict, out_dir: Path) -> None:
    df = pd.DataFrame(metrics)
    tsv_path = out_dir / "training_metrics.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Metrics saved to {tsv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["epoch"], df["constraint_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Constraint loss")
    axes[0].set_title("Constraint loss")

    axes[1].plot(df["epoch"], df["kl_loss"], label="Sampled KL")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL loss")
    axes[1].set_title("KL loss")
    axes[1].legend()

    if "validity" in df:
        axes[2].plot(df["epoch"], df["validity"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Validity")
    axes[2].set_title("Batch validity")
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    pdf_path = out_dir / "training_curves.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Training curves saved to {pdf_path}")


def sample_and_save(
    model: G2PTModel,
    base_model: G2PTModel,
    model_name: str,
    n_samples: int,
    batch_size: int,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    hstar: torch.Tensor,
    out_dir: Path,
    presample_z: torch.Tensor,
) -> None:
    model.eval()
    rows: list[dict[str, str | bool | float]] = []
    z_batches: list[torch.Tensor] = []

    pbar = tqdm(total=n_samples, desc="Sampling")
    n_done = 0
    while n_done < n_samples:
        n_batch = min(batch_size, n_samples - n_done)
        with torch.no_grad():
            samples = model.sample(n_batch)
            lp_ft = model.log_p(samples)
            lp_base = base_model.log_p(samples)
        smiles = decode_token_ids_to_smiles(
            samples.token_ids,
            model.tokenizer,
            model_name,
        )
        df, raw = descriptor_panel(smiles)
        z = standardize_descriptors(raw, df, mu, sigma)
        z_batches.append(z)

        for i, row in enumerate(df.itertuples(index=False)):
            rows.append(
                {
                    "smiles": row.smiles,
                    "valid": bool(row.valid),
                    "MolWt": float(row.MolWt) if row.valid else np.nan,
                    "MolLogP": float(row.MolLogP) if row.valid else np.nan,
                    "MolWt_z": z[i, 0].item(),
                    "MolLogP_z": z[i, 1].item(),
                    "log_p_finetuned": lp_ft[i].item(),
                    "log_p_base": lp_base[i].item(),
                }
            )

        n_done += n_batch
        pbar.update(n_batch)
    pbar.close()

    df = pd.DataFrame(rows)
    csv_path = out_dir / "samples.csv"
    df.to_csv(csv_path, index=False)
    kl_est = (df["log_p_finetuned"] - df["log_p_base"]).mean()
    validity = df["valid"].mean()
    print(f"Saved {n_samples} samples to {csv_path}")
    print(f"  Estimated KL(finetuned || base) = {kl_est:.3f} nats")
    print(f"  Validity = {validity:.3f}")

    valid_mask = df["valid"].to_numpy(dtype=bool)
    z_samples = torch.cat(z_batches, dim=0).numpy()[valid_mask]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        presample_z[:, 0].numpy(),
        presample_z[:, 1].numpy(),
        s=8,
        alpha=0.1,
        label="Base-model presamples",
    )
    ax.scatter(
        hstar[:, 0].numpy(),
        hstar[:, 1].numpy(),
        s=8,
        alpha=0.35,
        label="Target smiley",
    )
    ax.scatter(
        z_samples[:, 0],
        z_samples[:, 1],
        s=8,
        alpha=0.35,
        label="Model samples",
    )
    ax.set_xlabel("MolWt (standardized)")
    ax.set_ylabel("MolLogP (standardized)")
    ax.set_title("Smiley target vs model samples")
    ax.legend()
    ax.grid(alpha=0.2)
    ax.set_aspect("equal")
    fig.tight_layout()
    pdf_path = out_dir / "smiley_overlay.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Overlay plot saved to {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate G2PT toward a synthetic smiley distribution in the "
            "standardized (MolWt, MolLogP) plane."
        )
    )
    parser.add_argument("--lambd", type=float, default=0.1)
    parser.add_argument(
        "--loss_weighting",
        choices=["raw", "normalized"],
        default="raw",
        help="Whether lambd sets raw KL weight or a normalized KL:constraint ratio.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--cosine_schedule",
        action="store_true",
        help="Use linear warmup then cosine LR decay to lr/100.",
    )
    parser.add_argument(
        "--self_repulsion_weight",
        type=float,
        default=1.0,
        help="Coefficient on the model-model kernel term inside the MMD estimator.",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None,
        help="Optional global gradient-norm clipping threshold.",
    )
    parser.add_argument(
        "--batch_chunks",
        type=int,
        default=2,
        help="Chunks for log_p accumulation.",
    )
    parser.add_argument(
        "--n_hstar",
        type=int,
        default=10_000,
        help="Number of synthetic smiley samples used as the target distribution.",
    )
    parser.add_argument(
        "--n_norm",
        type=int,
        default=20_000,
        help=(
            "Number of base-model generations used to fit descriptor "
            "standardization and saved for plotting."
        ),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root directory containing shared cache files and per-run outputs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=20_480,
        help="Molecules to sample after training for evaluation.",
    )
    parser.add_argument(
        "--sample_every_epochs",
        type=int,
        default=0,
        help="Save seeded training sample snapshots every N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--n_training_samples",
        type=int,
        default=2048,
        help="Molecules to save in each training sample snapshot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use CUDA bfloat16 autocast for G2PT forward passes.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="xchen16/g2pt-moses-small-bfs",
        help="HuggingFace model ID for the pretrained G2PT model.",
    )
    parser.add_argument(
        "--kernel",
        choices=["energy", "rbf"],
        default="rbf",
        help="Kernel for the smiley calibration objective.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    training_sample_seed = args.seed

    out_root = Path(args.out_root)
    cache_dir = out_root / "cache"
    device = args.device
    warmup_epochs = (
        max(10, int(round(0.05 * args.epochs))) if args.cosine_schedule else 0
    )
    warmup_epochs = min(warmup_epochs, max(args.epochs - 1, 0))
    min_lr_ratio = 0.01
    run_name = (
        f"feature-molwt_logp-smiley_kernel-{args.kernel}"
        f"_lambd-{args.lambd}_selfwt-{args.self_repulsion_weight}_seed-{args.seed}"
    )
    out_dir = out_root / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading G2PT model...")
    model = G2PTModel(model_name=args.model_name, device=device, use_bf16=args.bf16)
    mu, sigma, n_valid_norm, presampled_norm_path, presample_z = (
        presample_base_model_for_standardization(
            model=model,
            model_name=args.model_name,
            n_samples=args.n_norm,
            batch_size=args.batch_size,
            out_dir=out_dir,
        )
    )
    model.train()

    hstar = sample_smiley_face(args.n_hstar, seed=args.seed)
    h = make_h(model.tokenizer, args.model_name, mu, sigma)
    if args.kernel == "energy":
        kernel = scale_kernel(energy_distance_kernel(), 10.0)
    elif args.kernel == "rbf":
        kernel = scale_kernel(rbf_mixture_kernel([0.05]), 100.0)
    else:
        raise ValueError(f"Unknown kernel {args.kernel}")

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "out_root": str(out_root),
                "cache_dir": str(cache_dir),
                "feature": "molwt_logp_smiley",
                "lambd": args.lambd,
                "loss_weighting": args.loss_weighting,
                "seed": args.seed,
                "lr": args.lr,
                "self_repulsion_weight": args.self_repulsion_weight,
                "cosine_schedule": args.cosine_schedule,
                "warmup_epochs": warmup_epochs,
                "min_lr_ratio": min_lr_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "bf16": args.bf16,
                "model_name": args.model_name,
                "kernel": args.kernel,
                "standardization_source": "base_model_samples",
                "n_hstar": args.n_hstar,
                "n_norm": args.n_norm,
                "n_valid_norm": n_valid_norm,
                "standardization_mean": mu.tolist(),
                "standardization_std": sigma.tolist(),
                "presampled_norm_samples_path": str(presampled_norm_path),
                "invalid_descriptor_value": [0.0, 0.0],
                "hole_radius": 0.35,
                "sample_every_epochs": args.sample_every_epochs,
                "n_training_samples": args.n_training_samples,
                "training_sample_seed": training_sample_seed,
            },
            f,
            indent=2,
        )

    pd.DataFrame(
        {
            "MolWt_z": hstar[:, 0].numpy(),
            "MolLogP_z": hstar[:, 1].numpy(),
        }
    ).to_csv(out_dir / "hstar_smiley.csv", index=False)

    base_model = utils.clone_network(model)
    dict_logger = utils.DictLogger()
    print(
        "Calibrating with feature=molwt_logp_smiley, "
        f"lambd={args.lambd}, loss_weighting={args.loss_weighting}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, "
        f"self_repulsion_weight={args.self_repulsion_weight}"
    )
    if args.epochs > 0:
        calibrate_mmd(
            model=model,
            h=h,
            hstar=hstar.to(device=device),
            lambd=args.lambd,
            loss_weighting=args.loss_weighting,
            kernel=kernel,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer_params={"lr": args.lr},
            cosine_schedule=args.cosine_schedule,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
            batch_chunks=args.batch_chunks,
            self_repulsion_weight=args.self_repulsion_weight,
            grad_clip_norm=args.grad_clip_norm,
            logger=make_logger(
                model.tokenizer,
                args.model_name,
                mu,
                sigma,
                dict_logger,
                out_dir,
                args.sample_every_epochs,
                args.n_training_samples,
                training_sample_seed,
                args.batch_size,
            ),
        )

        ckpt_path = out_dir / "final_checkpoint.pth"
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")

        save_results(dict_logger.metrics, out_dir)

    if args.n_eval_samples > 0:
        sample_and_save(
            model=model,
            base_model=base_model,
            model_name=args.model_name,
            n_samples=args.n_eval_samples,
            batch_size=args.batch_size,
            mu=mu,
            sigma=sigma,
            hstar=hstar,
            out_dir=out_dir,
            presample_z=presample_z,
        )


if __name__ == "__main__":
    main()
