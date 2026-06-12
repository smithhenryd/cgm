"""
KL-regularised finetuning of G2PT on a target SMILES CSV.

Loss = − E_data[log p_θ(x)] + λ · KL(θ ‖ base)

The cross-entropy term uses teacher-forced forward passes on mini-batches
drawn from --target_csv.  The KL term is estimated via sampling from the
current model, using the same REINFORCE estimator as calibrate_g2pt.py.
Each epoch is one gradient step (consistent with calibrate_g2pt.py).

Only molecules whose atoms/bonds fall within the MOSES G2PT vocabulary
(C, N, S, O, F, Cl, Br, H / single, double, triple, aromatic) are used.

Usage:
    PYTHONPATH=.. /Users/ndiamant/miniforge3/envs/g2pt/bin/python finetune_g2pt.py \\
        --target_csv abx_smiles.csv \\
        --out_root /path/to/output_root \\
        --lambd 0.1
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from g2pt_cgm_model import G2PTModel, G2PTSample
from features import tokens_to_smiles
from datasets_utils import to_seq_by_bfs
from cgm import utils
from cgm.lr_schedules import make_warmup_cosine_scheduler

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# GuacaMol atom/bond vocabulary — must match the pretrained g2pt-guacamol model.
_ATOM_TYPES = [
    "ATOM_C",
    "ATOM_N",
    "ATOM_O",
    "ATOM_F",
    "ATOM_B",
    "ATOM_Br",
    "ATOM_Cl",
    "ATOM_I",
    "ATOM_P",
    "ATOM_S",
    "ATOM_Se",
    "ATOM_Si",
]
_BOND_TYPES = ["BOND_SINGLE", "BOND_DOUBLE", "BOND_TRIPLE", "BOND_AROMATIC"]
_ATOM_DECODER: dict[str, int] = {
    sym: i
    for i, sym in enumerate(
        ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
    )
}
_BOND_DECODER: dict[BT, int] = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
}


def smiles_to_pyg(smi: str) -> Data | None:
    """
    Convert a SMILES string to a PyG Data object with one-hot node/edge features
    in the MOSES vocabulary. Returns None if the molecule is invalid or contains
    out-of-vocabulary atoms or bond types.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    type_idx: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in _ATOM_DECODER:
            return None
        type_idx.append(_ATOM_DECODER[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt not in _BOND_DECODER:
            return None
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [_BOND_DECODER[bt] + 1]

    if not row:
        return None

    N = mol.GetNumAtoms()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type_t, num_classes=len(_BOND_DECODER) + 1).float()
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]
    x = F.one_hot(torch.tensor(type_idx), num_classes=len(_ATOM_DECODER)).float()
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def pyg_batch_to_sample(data_list: list[Data], tokenizer) -> G2PTSample:
    """
    Encode a list of PyG Data objects to a padded G2PTSample.
    Each molecule gets a fresh random BFS node ordering (data augmentation).
    """
    texts = [to_seq_by_bfs(d, _ATOM_TYPES, _BOND_TYPES)["text"][0] for d in data_list]
    enc = tokenizer(texts, padding="max_length", return_tensors="pt", truncation=True)
    return G2PTSample(token_ids=enc["input_ids"])


def sample_and_save(
    model: G2PTModel,
    base_model: G2PTModel,
    n_samples: int,
    batch_size: int,
    out_dir: Path,
) -> None:
    """Sample from the finetuned model and save to out_dir/samples.csv."""
    model.eval()
    rows = []
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


def save_results(metrics: dict, out_dir: Path) -> None:
    """Save per-epoch metrics as a TSV and a two-panel PDF."""
    df = pd.DataFrame(metrics)
    tsv_path = out_dir / "training_metrics.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"Metrics saved to {tsv_path}")

    fig, (ax_ce, ax_kl) = plt.subplots(1, 2, figsize=(10, 4))
    ax_ce.plot(df["epoch"], df["ce_loss"])
    ax_ce.set_xlabel("Epoch")
    ax_ce.set_ylabel("CE loss")
    ax_ce.set_title("Cross-entropy loss")
    ax_kl.plot(df["epoch"], df["kl_loss"])
    ax_kl.set_xlabel("Epoch")
    ax_kl.set_ylabel("KL loss")
    ax_kl.set_title("KL loss")
    fig.tight_layout()
    pdf_path = out_dir / "training_curves.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Training curves saved to {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KL-regularised finetuning of G2PT on a target SMILES CSV."
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        required=True,
        help="CSV with a SMILES column to finetune on.",
    )
    parser.add_argument(
        "--lambd", type=float, default=0.1, help="KL regularization strength."
    )
    parser.add_argument(
        "--loss_weighting",
        choices=["raw", "normalized"],
        default="raw",
        help="Whether lambd sets raw KL weight or a normalized KL:CE ratio.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Molecules per step."
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
        help="Chunks for training log_p calls (reduces peak GPU memory).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Root directory for per-run outputs.",
    )
    parser.add_argument("--seed", type=int, default=0)
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
        "--model_name",
        type=str,
        default="xchen16/g2pt-moses-small-bfs",
        help="HuggingFace model ID for the pretrained G2PT model.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_root = Path(args.out_root)
    device = args.device
    warmup_epochs = (
        max(10, int(round(0.05 * args.epochs))) if args.cosine_schedule else 0
    )
    warmup_epochs = min(warmup_epochs, max(args.epochs - 1, 0))
    min_lr_ratio = 0.01

    # ---- Load model ----
    print("Loading G2PT model...")
    model = G2PTModel(model_name=args.model_name, device=device, use_bf16=args.bf16)
    model.train()

    # ---- Load and preprocess target SMILES ----
    # The tokenizer uses a fixed WordLevel vocab with no UNK token; IDX_N tokens
    # only exist up to the largest molecule in the training set.  Molecules with
    # more atoms than that would produce unknown tokens and crash tokenization.
    max_atoms = (
        max(int(t.split("_")[1]) for t in model.tokenizer.vocab if t.startswith("IDX_"))
        + 1
    )

    all_smiles = pd.read_csv(args.target_csv)["SMILES"].tolist()
    print(f"Loaded {len(all_smiles)} SMILES from {args.target_csv}")

    pyg_data: list[Data] = []
    skipped: list[str] = []
    for smi in all_smiles:
        d = smiles_to_pyg(smi)
        if d is not None and d.num_nodes <= max_atoms:
            pyg_data.append(d)
        else:
            skipped.append(smi)

    print(
        f"  {len(pyg_data)} molecules retained, "
        f"{len(skipped)} skipped (invalid, out-of-vocab atoms/bonds, or too large)"
    )
    print(f"  Tokenizer max atom count: {max_atoms}")
    if skipped:
        preview = skipped[:3]
        print(f"  Skipped examples: {preview}{'...' if len(skipped) > 3 else ''}")
    if not pyg_data:
        raise ValueError(
            "No valid molecules found in target CSV for MOSES G2PT vocabulary."
        )

    # ---- Output directory ----
    target_stem = Path(args.target_csv).stem
    run_name = f"target-{target_stem}_lambd-{args.lambd}_seed-{args.seed}"
    out_dir = out_root / "finetune_runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(
            {
                "out_root": str(out_root),
                "target_csv": args.target_csv,
                "n_target_molecules": len(pyg_data),
                "n_skipped": len(skipped),
                "lambd": args.lambd,
                "loss_weighting": args.loss_weighting,
                "seed": args.seed,
                "lr": args.lr,
                "cosine_schedule": args.cosine_schedule,
                "warmup_epochs": warmup_epochs,
                "min_lr_ratio": min_lr_ratio,
                "grad_clip_norm": args.grad_clip_norm,
                "batch_chunks": args.batch_chunks,
                "bf16": args.bf16,
                "model_name": args.model_name,
            },
            f,
            indent=2,
        )

    # ---- Clone base model before training ----
    base_model = utils.clone_network(model)

    # ---- Optimizer + LR schedule ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.cosine_schedule:
        scheduler = make_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=args.epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        scheduler = None

    # ---- Training loop ----
    loss_weighting: Literal["raw", "normalized"] = args.loss_weighting
    if loss_weighting == "raw":
        ce_weight = 1.0
        kl_weight = args.lambd
    else:
        denom = 1.0 + args.lambd
        ce_weight = 1.0 / denom
        kl_weight = args.lambd / denom

    n_data = len(pyg_data)
    dict_logger = utils.DictLogger()
    print(
        f"Finetuning: target={args.target_csv}, lambd={args.lambd}, "
        f"loss_weighting={loss_weighting}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )

    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        # Sample a mini-batch from target data (with replacement)
        idx = torch.randint(0, n_data, (args.batch_size,))
        batch_data = [pyg_data[i] for i in idx.tolist()]
        target_sample = pyg_batch_to_sample(batch_data, model.tokenizer)
        target_sample = G2PTSample(token_ids=target_sample.token_ids.to(device))

        # Sample from model for KL estimate
        with torch.no_grad():
            xs = model.sample(args.batch_size)

        optimizer.zero_grad(set_to_none=True)

        # CE loss: teacher-forced, gradient flows directly
        ce_loss = -model.log_p(target_sample, batch_chunks=args.batch_chunks).mean()
        (ce_weight * ce_loss).backward()

        # KL REINFORCE coefficients (no grad needed for the coefficient values)
        with torch.no_grad():
            log_p_theta_coeff = model.log_p(xs, batch_chunks=args.batch_chunks)
            log_p_base_vals = base_model.log_p(xs, batch_chunks=args.batch_chunks)
        kl_coeffs = (log_p_theta_coeff - log_p_base_vals) / args.batch_size

        # REINFORCE: gradient flows through this separate log_p call
        kl_reinforce_loss = (
            kl_coeffs.detach() * model.log_p(xs, batch_chunks=args.batch_chunks)
        ).sum()
        (kl_weight * kl_reinforce_loss).backward()

        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        kl_est = (log_p_theta_coeff - log_p_base_vals).mean().item()
        scalars = {
            "epoch": epoch,
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_est,
            "loss_weighting": loss_weighting,
            "ce_weight": ce_weight,
            "kl_weight": kl_weight,
        }
        dict_logger(scalars)
        utils.default_logger(scalars)

    # ---- Save checkpoint ----
    if args.epochs > 0:
        ckpt_path = out_dir / "final_checkpoint.pth"
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")
        save_results(dict_logger.metrics, out_dir)

    # ---- Evaluate ----
    if args.n_eval_samples > 0:
        sample_and_save(
            model, base_model, args.n_eval_samples, args.batch_size, out_dir
        )


if __name__ == "__main__":
    main()
