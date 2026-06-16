from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import pandas as pd
import torch
from tqdm import tqdm

from cgm import utils
from cgm.cgm_distribution import fit_pca
from features import (
    KCGM_DESCRIPTOR_COLUMNS,
    fcd_features,
    fit_descriptor_scaling,
    load_fcd_model,
    load_moses_split_smiles,
    morgan_features,
    murcko_scaffold_smiles,
    murcko_scaffolds,
    sigma_scaled_descriptor_features,
    tokens_to_smiles,
)
from g2pt_cgm_model import G2PTModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class FeatureSetup:
    feature: str
    h: Callable[[Any], torch.Tensor]
    target_features: torch.Tensor
    metadata: dict[str, Any]


def load_target_smiles(cache_dir: Path, target_csv: str | None) -> list[str]:
    if target_csv is not None:
        return pd.read_csv(target_csv)["SMILES"].tolist()
    return load_moses_split_smiles(cache_dir, split="train")


def subsample_target_smiles(
    all_smiles: list[str],
    n_hstar: int,
) -> list[str]:
    n_target = min(n_hstar, len(all_smiles))
    idx = torch.randperm(len(all_smiles))[:n_target]
    return [all_smiles[i] for i in idx.tolist()]


def preprocess_target_smiles(feature: str, target_smiles: list[str]) -> list[str]:
    """
    Apply feature-specific filtering to the empirical target sample before
    feature construction and provenance saving.
    """
    if feature == "generic_murcko_fcd":
        return [
            smi
            for smi in target_smiles
            if murcko_scaffold_smiles(smi, generic=True) is not None
        ]
    return target_smiles


def build_feature_setup(
    *,
    feature: str,
    target_smiles: list[str],
    tokenizer: Any,
    device: str,
    n_pca: int,
) -> FeatureSetup:
    if feature == "morgan":
        target_features = morgan_features(target_smiles)
        h = lambda x: morgan_features(tokens_to_smiles(x.token_ids, tokenizer)).to(
            device
        )
        return FeatureSetup(
            feature=feature,
            h=h,
            target_features=target_features,
            metadata={},
        )

    if feature == "fcd":
        fcd_model = load_fcd_model(device=device)
        target_raw = fcd_features(target_smiles, fcd_model)
        pca = fit_pca(target_raw, n_components=n_pca, whiten=False)
        h_raw = lambda x: fcd_features(
            tokens_to_smiles(x.token_ids, tokenizer),
            fcd_model,
        )
        h = lambda x: pca(h_raw(x)).to(device)
        return FeatureSetup(
            feature=feature,
            h=h,
            target_features=pca(target_raw),
            metadata={"n_pca": n_pca},
        )

    if feature == "descriptors":
        sigma, n_valid = fit_descriptor_scaling(
            target_smiles,
            descriptor_columns=KCGM_DESCRIPTOR_COLUMNS,
        )
        target_features = sigma_scaled_descriptor_features(
            target_smiles,
            sigma,
            descriptor_columns=KCGM_DESCRIPTOR_COLUMNS,
        )

        def h(x: Any) -> torch.Tensor:
            smiles = tokens_to_smiles(x.token_ids, tokenizer)
            return sigma_scaled_descriptor_features(
                smiles,
                sigma,
                descriptor_columns=KCGM_DESCRIPTOR_COLUMNS,
            ).to(device)

        return FeatureSetup(
            feature=feature,
            h=h,
            target_features=target_features,
            metadata={
                "descriptor_columns": list(KCGM_DESCRIPTOR_COLUMNS),
                "descriptor_scaling": "target_std_only",
                "descriptor_scale_fit_n_valid": n_valid,
                "invalid_descriptor_value": [0.0] * len(KCGM_DESCRIPTOR_COLUMNS),
            },
        )

    if feature == "generic_murcko_fcd":
        fcd_model = load_fcd_model(device=device)
        target_scaffolds = murcko_scaffolds(target_smiles, generic=True)
        target_raw = fcd_features(target_scaffolds, fcd_model)
        pca = fit_pca(target_raw, n_components=n_pca, whiten=False)

        def h_raw(x: Any) -> torch.Tensor:
            smiles = tokens_to_smiles(x.token_ids, tokenizer)
            scaffolds = murcko_scaffolds(smiles, generic=True)
            return fcd_features(scaffolds, fcd_model)

        h = lambda x: pca(h_raw(x)).to(device)
        return FeatureSetup(
            feature=feature,
            h=h,
            target_features=pca(target_raw),
            metadata={
                "n_pca": n_pca,
                "scaffold": "generic_murcko",
                "base_feature": "fcd",
            },
        )

    raise ValueError(f"Unknown feature {feature!r}")


def target_tag(target_csv: str | None) -> str:
    return f"target-{Path(target_csv).stem}_" if target_csv is not None else ""


def sample_and_save(
    model: G2PTModel,
    base_model: G2PTModel,
    n_samples: int,
    batch_size: int,
    out_dir: Path,
) -> None:
    """
    Sample from the finetuned model and save to out_dir/samples.csv.

    Columns:
        smiles          canonical SMILES (empty string if invalid)
        valid           whether the sequence parsed as a valid molecule
        log_p_finetuned per-sample log probability under the finetuned model
        log_p_base      per-sample log probability under the pretrained base model

    The difference log_p_finetuned - log_p_base is an unbiased per-sample
    estimate of KL(finetuned || base), useful for trading off constraint
    reduction against deviation from the base model.
    """
    model.eval()
    rows: list[dict[str, str | bool | float]] = []
    pbar = tqdm(total=n_samples, desc="Sampling")
    n_done = 0
    while n_done < n_samples:
        n_batch = min(batch_size, n_samples - n_done)
        with torch.no_grad():
            samples = model.sample(n_batch)
            lp_ft = model.log_p(samples)
            lp_base = base_model.log_p(samples)
        smiles = tokens_to_smiles(samples.token_ids, model.tokenizer)
        for i, smi in enumerate(smiles):
            rows.append(
                {
                    "smiles": smi if smi is not None else "",
                    "valid": smi is not None,
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
    print(f"Saved {n_samples} samples to {csv_path}")
    print(f"  Estimated KL(finetuned || base) = {kl_est:.3f} nats")
    print(f"  Validity = {df['valid'].mean():.3f}")


def make_training_logger(dict_logger: utils.DictLogger) -> Callable:
    def logger(logs: dict, model: G2PTModel, base_model: G2PTModel, xs) -> None:
        scalars = {k: v for k, v in logs.items() if k != "h_bar"}
        dict_logger(scalars)
        utils.default_logger(scalars)

    return logger


def save_training_results(
    metrics: dict[str, list[float]],
    out_dir: Path,
    *,
    constraint_label: str,
) -> None:
    df = pd.DataFrame(metrics)
    tsv_path = out_dir / "training_metrics.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Metrics saved to {tsv_path}")

    fig, (ax_constraint, ax_kl) = plt.subplots(1, 2, figsize=(10, 4))

    ax_constraint.plot(df["epoch"], df["constraint_loss"])
    ax_constraint.set_xlabel("Epoch")
    ax_constraint.set_ylabel(constraint_label)
    ax_constraint.set_title("Constraint loss")

    ax_kl.plot(df["epoch"], df["kl_loss"], label="Sampled KL")
    if "kl_loss_exact" in df:
        ax_kl.plot(df["epoch"], df["kl_loss_exact"], label="Exact conditional KL")
    ax_kl.set_xlabel("Epoch")
    ax_kl.set_ylabel("KL loss")
    ax_kl.set_title("KL loss")
    ax_kl.legend()

    fig.tight_layout()
    pdf_path = out_dir / "training_curves.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Training curves saved to {pdf_path}")
