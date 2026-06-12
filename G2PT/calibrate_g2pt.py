"""
Calibrate a pretrained G2PT model to match the MOSES train-set distribution
via MMD, using either:
  - Morgan fingerprints + Tanimoto or dot-product kernel (no PCA needed)
  - FCD (ChemNet) activations + energy-distance kernel + PCA whitening
  - Target-scaled RDKit descriptors + energy-distance, RBF, or dot-product kernel
  - Generic Murcko scaffold FCD + energy-distance, RBF, or dot-product kernel

Usage:
    PYTHONPATH=.. /Users/ndiamant/miniforge3/envs/g2pt/bin/python calibrate_g2pt.py \\
        --out_root /path/to/output_root \\
        --feature morgan \\
        --lambd 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import pandas as pd

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from cgm import utils
from g2pt_cgm_model import G2PTModel
from cgm.cgm_distribution import (
    calibrate_mmd,
    tanimoto_kernel,
    energy_distance_kernel,
    rbf_mixture_kernel,
    dot_product_kernel,
)
from g2pt_calibration_common import (
    build_feature_setup,
    load_target_smiles,
    make_training_logger,
    preprocess_target_smiles,
    sample_and_save,
    save_training_results,
    subsample_target_smiles,
    target_tag,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate G2PT to match the MOSES train-set feature distribution."
    )
    parser.add_argument(
        "--feature",
        choices=["morgan", "fcd", "descriptors", "generic_murcko_fcd"],
        default="morgan",
        help="Molecular feature type for MMD calibration.",
    )
    parser.add_argument(
        "--lambd", type=float, default=0.1, help="KL regularization strength."
    )
    parser.add_argument(
        "--loss_weighting",
        choices=["raw", "normalized"],
        default="raw",
        help="Whether lambd sets raw KL weight or a normalized KL:constraint ratio.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Molecules sampled per step."
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--cosine_schedule",
        action="store_true",
        help="Use linear warmup then cosine LR decay to lr/100.",
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
        default=1,
        help="Chunks for log_p accumulation (reduces peak GPU memory).",
    )
    parser.add_argument(
        "--n_hstar",
        type=int,
        default=10_000,
        help="Subsample this many train molecules as target hstar.",
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=64,
        help=(
            "PCA components for FCD-derived features "
            "(ignored when --feature is morgan or descriptors)."
        ),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root directory containing shared cache files and per-run outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global RNG seed (controls hstar subsampling, training sampling, and eval sampling).",
    )
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=0,
        help="Molecules to sample after training for evaluation (0 = skip).",
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
        "--kernel",
        type=str,
        choices=["tanimoto", "energy", "rbf", "dot"],
        required=True,
        help="Kernel for MMD.",
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        default=None,
        help="CSV with a SMILES column to use as calibration target instead of MOSES train.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="xchen16/g2pt-moses-small-bfs",
        help="HuggingFace model ID for the pretrained G2PT model.",
    )
    parser.add_argument(
        "--no_loo",
        action="store_true",
        help=(
            "Disable leave-one-out correction for the MMD coefficients. "
            "KL centering is still applied."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_root = Path(args.out_root)
    cache_dir = out_root / "cache"
    device = args.device
    warmup_epochs = (
        max(10, int(round(0.05 * args.epochs))) if args.cosine_schedule else 0
    )
    warmup_epochs = min(warmup_epochs, max(args.epochs - 1, 0))
    min_lr_ratio = 0.01

    # ---- Load pretrained G2PT model ----
    print("Loading G2PT model...")
    model = G2PTModel(model_name=args.model_name, device=device, use_bf16=args.bf16)
    model.train()

    # ---- Load target SMILES and build features ----
    all_smiles = load_target_smiles(cache_dir, args.target_csv)
    hstar_smiles = subsample_target_smiles(all_smiles, args.n_hstar)
    hstar_n_before_filter = len(hstar_smiles)
    hstar_smiles = preprocess_target_smiles(args.feature, hstar_smiles)
    hstar_n_after_filter = len(hstar_smiles)
    feature_setup = build_feature_setup(
        feature=args.feature,
        target_smiles=hstar_smiles,
        tokenizer=model.tokenizer,
        device=device,
        n_pca=args.n_pca,
    )

    kernel_name = args.kernel
    match kernel_name:
        case "tanimoto":
            if args.feature != "morgan":
                raise ValueError(
                    "tanimoto kernel is only supported for morgan features, "
                    f"got feature={args.feature!r}"
                )
            kernel = tanimoto_kernel
        case "energy":
            kernel = energy_distance_kernel()
        case "rbf":
            kernel = rbf_mixture_kernel()
        case "dot":
            kernel = dot_product_kernel
        case _:
            raise ValueError(f"Unknown kernel {kernel_name}")

    # ---- Output directory ----
    run_target_tag = target_tag(args.target_csv)
    mmd_loo_tag = "off" if args.no_loo else "on"
    run_name = (
        f"{run_target_tag}feature-{args.feature}-{kernel_name}_mmdloo-{mmd_loo_tag}"
    )
    if args.feature not in {"morgan", "descriptors"}:
        run_name += f"_n_pca-{args.n_pca}"
    run_name += f"_lambd-{args.lambd}_seed-{args.seed}"
    out_dir = out_root / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata so evaluation can locate the shared cache.
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "out_root": str(out_root),
                "cache_dir": str(cache_dir),
                "feature": args.feature,
                "lambd": args.lambd,
                "loss_weighting": args.loss_weighting,
                "seed": args.seed,
                "kernel": kernel_name,
                "mmd_loo": not args.no_loo,
                "n_pca": args.n_pca,
                "lr": args.lr,
                "cosine_schedule": args.cosine_schedule,
                "warmup_epochs": warmup_epochs,
                "min_lr_ratio": min_lr_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "bf16": args.bf16,
                "target_csv": args.target_csv,
                "model_name": args.model_name,
                "objective": "mmd",
                "n_hstar_requested": args.n_hstar,
                "n_hstar_before_feature_filter": hstar_n_before_filter,
                "n_hstar_after_feature_filter": hstar_n_after_filter,
                **feature_setup.metadata,
            },
            f,
            indent=2,
        )

    # Save the hstar SMILES for provenance and novelty-vs-hstar evaluation.
    pd.DataFrame({"smiles": hstar_smiles}).to_csv(
        out_dir / "hstar_smiles.csv", index=False
    )

    # ---- Clone base model before training (needed for post-hoc log_p comparison) ----
    base_model = utils.clone_network(model)

    # ---- Calibrate ----
    dict_logger = utils.DictLogger()
    print(
        f"Calibrating with feature={args.feature}, kernel={kernel_name}, "
        f"mmd_loo={not args.no_loo}, lambd={args.lambd}, "
        f"loss_weighting={args.loss_weighting}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )
    if args.epochs > 0:  # useful to set epochs = 0 for getting samples from base model
        calibrate_mmd(
            model=model,
            h=feature_setup.h,
            hstar=feature_setup.target_features,
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
            grad_clip_norm=args.grad_clip_norm,
            logger=make_training_logger(dict_logger),
            use_loo=not args.no_loo,
            kernel_scale="auto",
        )

        # ---- Save final checkpoint ----
        ckpt_path = out_dir / "final_checkpoint.pth"
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")

        # ---- Save metrics and plots ----
        save_training_results(
            dict_logger.metrics,
            out_dir,
            constraint_label="Constraint loss (MMD)",
        )

    # ---- Sample and evaluate ----
    if args.n_eval_samples > 0:
        sample_and_save(
            model, base_model, args.n_eval_samples, args.batch_size, out_dir
        )


if __name__ == "__main__":
    main()
