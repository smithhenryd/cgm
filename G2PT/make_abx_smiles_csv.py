"""
Convert GNEprop_known_abx.tsv to a clean abx_smiles.csv with a single SMILES column.
Mixture records (dot-separated SMILES) are split into one row per fragment,
canonicalized without chirality, and exact duplicate canonical SMILES are
removed.

Each fragment is tested by running the full finetune_g2pt tokenization pipeline
(smiles_to_pyg → to_seq_by_bfs → tokenizer) against the GuacaMol G2PT model.
Only fragments that survive the pipeline without error are kept, ensuring that
calibrate_g2pt.py and finetune_g2pt.py use an identical target set.

Usage:
    python make_abx_smiles_csv.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

G2PT_DIR = Path(__file__).parent
sys.path.insert(0, str(G2PT_DIR))

from finetune_g2pt import smiles_to_pyg, _ATOM_TYPES, _BOND_TYPES
from datasets_utils import to_seq_by_bfs
from g2pt_cgm_model import G2PTModel

_MORGAN_GEN_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def _canonicalize_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _tokenizes_ok(smi: str, model: G2PTModel) -> bool:
    """Return True iff the molecule survives the full finetune_g2pt pipeline."""
    data = smiles_to_pyg(smi)
    if data is None:
        return False
    max_atoms = (
        max(int(t.split("_")[1]) for t in model.tokenizer.vocab if t.startswith("IDX_"))
        + 1
    )
    if data.num_nodes > max_atoms:
        return False
    try:
        text = to_seq_by_bfs(data, _ATOM_TYPES, _BOND_TYPES)["text"][0]
        enc = model.tokenizer(text, return_tensors="pt")
        if enc["input_ids"].shape[1] > model.tokenizer.model_max_length:
            return False
    except Exception:
        return False
    return True


def _print_pairwise_morgan_similarity(smiles: list[str]) -> None:
    if len(smiles) < 2:
        print("Pairwise Morgan similarity check skipped: fewer than 2 SMILES")
        return

    fps = [_MORGAN_GEN_1024.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]

    all_sims: list[float] = []
    max_sim = -1.0
    max_pair: tuple[int, int] | None = None
    for j in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[j], fps[:j])
        all_sims.extend(sims)
        i = int(np.argmax(sims))
        if sims[i] > max_sim:
            max_sim = float(sims[i])
            max_pair = (i, j)

    sim_arr = np.asarray(all_sims, dtype=float)
    print(
        "Pairwise Morgan Tanimoto (radius=2, fpSize=1024): "
        f"n_pairs={len(sim_arr)}, mean={sim_arr.mean():.4f}, "
        f"median={np.median(sim_arr):.4f}, p95={np.quantile(sim_arr, 0.95):.4f}, "
        f"max={sim_arr.max():.4f}"
    )
    if max_pair is not None:
        i, j = max_pair
        print(f"Most similar pair: {smiles[i]} || {smiles[j]}")


print("Loading GuacaMol G2PT model for tokenization check...")
model = G2PTModel(model_name="xchen16/g2pt-guacamol-small-bfs", device="cpu")

df = pd.read_csv(G2PT_DIR / "GNEprop_known_abx.tsv", sep="\t")

smiles, skipped = [], []
for smi in df["SMILES"]:
    for frag in smi.split("."):
        canonical = _canonicalize_smiles(frag)
        if canonical is None:
            skipped.append(frag)
        elif _tokenizes_ok(canonical, model):
            smiles.append(canonical)
        else:
            skipped.append(canonical)

out = pd.DataFrame({"SMILES": smiles}).drop_duplicates(ignore_index=True)
out_path = G2PT_DIR / "abx_smiles.csv"
out.to_csv(out_path, index=False)
print(f"Wrote {len(out)} SMILES to {out_path} ({len(skipped)} skipped)")
_print_pairwise_morgan_similarity(out["SMILES"].tolist())
