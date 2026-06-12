"""
Feature extractors for G2PT calibration.

Core feature maps:
  - Morgan fingerprints (radius=2, 256 bits) with Tanimoto kernel
  - ChemNet activations (512-dim, via fcd_torch) with energy-distance kernel

Descriptor helpers are also provided for target-scaled RDKit feature spaces.
All feature functions return float tensors with zero vectors for
invalid/unparseable molecules.
"""

import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import fcd_torch
from functools import partial
from rdkit import Chem, RDConfig
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

from rdkit import RDLogger

# Disable standard error and warning messages
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

sys.path.insert(0, str(Path(__file__).parent))
from datasets_utils import seq_to_mol, get_smiles

try:
    from rdkit.Contrib.SA_Score import sascorer
except ImportError:
    sa_score_dir = Path(RDConfig.RDContribDir) / "SA_Score"
    if str(sa_score_dir) not in sys.path:
        sys.path.append(str(sa_score_dir))
    import sascorer

_MOSES_SPLIT_URLS = {
    "train": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/train.csv",
    "val": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test.csv",
    "test": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/test_scaffolds.csv",
}


# ---------------------------------------------------------------------------
# Token → SMILES
# ---------------------------------------------------------------------------
def to_smiles(seq_str: str) -> str | None:
    try:
        mol = seq_to_mol(seq_str)
        return get_smiles(mol)  # None if mol is invalid
    except Exception:
        return None


def tokens_to_smiles(
    token_ids: torch.Tensor,  # [N, T]
    tokenizer,
    max_workers: int = 4,
) -> list[str | None]:
    """
    Decode padded token sequences to canonical SMILES strings.
    Returns None for sequences that fail to parse as valid molecules.
    """
    seq_strs = tokenizer.batch_decode(token_ids)

    if max_workers == 1:
        return [to_smiles(seq_str) for seq_str in seq_strs]

    sm = process_map(
        to_smiles,
        seq_strs,
        max_workers=max_workers,
        disable=True,
        chunksize=512,
    )
    return sm


# ---------------------------------------------------------------------------
# Morgan fingerprints
# ---------------------------------------------------------------------------


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)


def _fp_one(smi: str | None, n_bits: int = 256) -> np.ndarray:
    """Worker function to process a single SMILES string."""
    mol = Chem.MolFromSmiles(smi) if smi is not None else None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)

    fp = _MORGAN_GEN.GetFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def morgan_features(
    smiles_list: list[str | None],
    n_bits: int = 256,
    max_workers: int = 4,
) -> torch.Tensor:
    """
    Compute Morgan fingerprints (radius=2) for each SMILES.
    Invalid or None entries return zero vectors.

    Args:
        smiles_list: SMILES strings (None entries treated as invalid).
        n_bits:      Fingerprint length.
        max_workers: Worker processes (None = all CPUs, 1 = sequential).

    Returns:
        [N, n_bits] float32 tensor suitable for use with tanimoto_kernel.
    """
    fps = process_map(
        partial(_fp_one, n_bits=n_bits),
        smiles_list,
        max_workers=max_workers,
        desc="Morgan FP",
        chunksize=512,
        leave=False,
    )
    return torch.from_numpy(np.stack(fps))


# ---------------------------------------------------------------------------
# FCD (ChemNet) features
# ---------------------------------------------------------------------------

_FCD_DIM = 512
RDKIT_DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
]
EMBEDDING_DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHAcceptors",
    "NumHDonors",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
]
KCGM_DESCRIPTOR_COLUMNS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "QED",
    "SAScore",
    "FractionCSP3",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "HeavyAtomCount",
]


def _sa_score(mol: Chem.Mol) -> float:
    return float(sascorer.calculateScore(mol))


_DESCRIPTOR_EXTRACTORS = {
    "MolWt": lambda mol: float(Descriptors.MolWt(mol)),
    "MolLogP": lambda mol: float(Crippen.MolLogP(mol)),
    "TPSA": lambda mol: float(rdMolDescriptors.CalcTPSA(mol)),
    "NumRotatableBonds": lambda mol: float(Lipinski.NumRotatableBonds(mol)),
    "NumHAcceptors": lambda mol: float(Lipinski.NumHAcceptors(mol)),
    "NumHDonors": lambda mol: float(Lipinski.NumHDonors(mol)),
    "RingCount": lambda mol: float(rdMolDescriptors.CalcNumRings(mol)),
    "NumAromaticRings": lambda mol: float(rdMolDescriptors.CalcNumAromaticRings(mol)),
    "FractionCSP3": lambda mol: float(rdMolDescriptors.CalcFractionCSP3(mol)),
    "HeavyAtomCount": lambda mol: float(mol.GetNumHeavyAtoms()),
    "QED": lambda mol: float(QED.qed(mol)),
    "SAScore": _sa_score,
}


def load_fcd_model(device: str = "cpu") -> fcd_torch.FCD:
    """Load the ChemNet model used by FCD. Call once and reuse."""
    return fcd_torch.FCD(device=device, n_jobs=1)


def fcd_features(
    smiles_list: list[str | None],
    fcd_model: Optional[fcd_torch.FCD] = None,
) -> torch.Tensor:
    """
    Compute ChemNet activations (512-dim) for each SMILES.
    Invalid or None entries return zero vectors.
    fcd_model is created on CPU if not provided; pass a pre-loaded model
    to avoid reloading ChemNet on every call.

    Returns:
        [N, 512] float32 tensor suitable for use with energy_distance_kernel.
    """
    if fcd_model is None:
        fcd_model = load_fcd_model()

    valid_idx: list[int] = []
    valid_smiles: list[str] = []
    for i, smi in enumerate(smiles_list):
        canonical = canonicalize_smiles(smi)
        if canonical is None:
            continue
        valid_idx.append(i)
        valid_smiles.append(canonical)

    out = torch.zeros(len(smiles_list), _FCD_DIM, dtype=torch.float32)
    if valid_smiles:
        acts = fcd_model.get_predictions(valid_smiles)  # [n_valid, 512] numpy
        out[valid_idx] = torch.from_numpy(acts)
    return out


def canonicalize_smiles(smi: str | None) -> str | None:
    """Return canonical RDKit SMILES, or None if parsing/sanitization fails."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def murcko_scaffold_smiles(smi: str | None, *, generic: bool = False) -> str | None:
    """
    Return the canonical Bemis-Murcko scaffold SMILES for one molecule.

    Molecules without a non-empty scaffold return None so they do not all collapse
    to the same empty-string scaffold.
    """
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None
    if generic:
        try:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        except Chem.rdchem.AtomValenceException:
            scaffold = _make_scaffold_generic_fallback(scaffold)

    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return canonicalize_smiles(scaffold_smiles)


def murcko_scaffolds(
    smiles_list: list[str | None],
    *,
    generic: bool = False,
) -> list[str | None]:
    """Extract exact or generic Bemis-Murcko scaffolds for a SMILES list."""
    return [murcko_scaffold_smiles(smi, generic=generic) for smi in smiles_list]


def _make_scaffold_generic_fallback(scaffold: Chem.Mol) -> Chem.Mol:
    """
    RDKit's MakeScaffoldGeneric occasionally fails on valid scaffolds during H
    removal. This fallback performs the same atom/bond rewriting but avoids the
    problematic post-processing step.
    """
    generic = Chem.Mol(scaffold)
    for atom in generic.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom.SetAtomicNum(6)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in generic.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return generic


def _descriptor_row(
    smi: str | None,
    descriptor_columns: tuple[str, ...],
) -> tuple[bool, np.ndarray]:
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return False, np.zeros(
            len(descriptor_columns),
            dtype=np.float32,
        )

    values = np.asarray(
        [_DESCRIPTOR_EXTRACTORS[name](mol) for name in descriptor_columns],
        dtype=np.float32,
    )
    return True, values


def _descriptor_rows(
    smiles_list: list[str | None],
    *,
    verbose: bool,
    descriptor_columns: Sequence[str] = EMBEDDING_DESCRIPTOR_COLUMNS,
) -> list[tuple[bool, np.ndarray]]:
    descriptor_columns_tuple = tuple(descriptor_columns)
    kwargs = {
        "max_workers": 4,
        "chunksize": 512,
        "disable": not verbose,
        "desc": "Descriptor rows",
        "leave": False,
    }
    try:
        return process_map(
            partial(_descriptor_row, descriptor_columns=descriptor_columns_tuple),
            smiles_list,
            **kwargs,
        )
    except PermissionError:
        return thread_map(
            partial(_descriptor_row, descriptor_columns=descriptor_columns_tuple),
            smiles_list,
            **kwargs,
        )


def fit_descriptor_standardization(
    smiles_list: Iterable[str | None],
    *,
    verbose: bool = False,
    descriptor_columns: Sequence[str] = EMBEDDING_DESCRIPTOR_COLUMNS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fit mean/std for the descriptor embedding on valid molecules only.

    Returns:
        mu: [D] descriptor mean
        sigma: [D] descriptor std (clamped away from zero)
        n_valid: count of valid molecules used for fitting
    """
    smiles_seq = list(smiles_list)
    rows = _descriptor_rows(
        smiles_seq,
        verbose=verbose,
        descriptor_columns=descriptor_columns,
    )
    valid_rows = [values for is_valid, values in rows if is_valid]
    n_valid = len(valid_rows)

    if n_valid == 0:
        raise ValueError("No valid molecules available to fit descriptor scaling.")

    valid_tensor = torch.from_numpy(np.stack(valid_rows))
    mu = valid_tensor.mean(dim=0)
    sigma = valid_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mu, sigma, n_valid


def standardized_descriptor_features(
    smiles_list: list[str | None],
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    verbose: bool = False,
    descriptor_columns: Sequence[str] = EMBEDDING_DESCRIPTOR_COLUMNS,
) -> torch.Tensor:
    """
    Standardize descriptor embeddings. Invalid molecules remain all zeros.
    """
    rows = _descriptor_rows(
        smiles_list,
        verbose=verbose,
        descriptor_columns=descriptor_columns,
    )
    out = torch.zeros((len(smiles_list), len(descriptor_columns)), dtype=torch.float32)
    for i, (is_valid, values) in enumerate(rows):
        if is_valid:
            out[i] = (torch.from_numpy(values) - mu) / sigma
    return out


def fit_descriptor_scaling(
    smiles_list: Iterable[str | None],
    *,
    verbose: bool = False,
    descriptor_columns: Sequence[str] = KCGM_DESCRIPTOR_COLUMNS,
) -> tuple[torch.Tensor, int]:
    """
    Fit per-dimension standard deviations on valid molecules only.

    This is intended for target-scaled descriptor spaces where features are
    divided by the target standard deviation without mean-centering, so the
    all-zero invalid vector remains far from the target cloud.
    """
    _, sigma, n_valid = fit_descriptor_standardization(
        smiles_list,
        verbose=verbose,
        descriptor_columns=descriptor_columns,
    )
    return sigma, n_valid


def sigma_scaled_descriptor_features(
    smiles_list: list[str | None],
    sigma: torch.Tensor,
    *,
    verbose: bool = False,
    descriptor_columns: Sequence[str] = KCGM_DESCRIPTOR_COLUMNS,
) -> torch.Tensor:
    """
    Scale descriptor embeddings by the target standard deviation only.

    Invalid molecules remain all zeros.
    """
    rows = _descriptor_rows(
        smiles_list,
        verbose=verbose,
        descriptor_columns=descriptor_columns,
    )
    out = torch.zeros(
        (len(smiles_list), len(descriptor_columns)),
        dtype=torch.float32,
    )
    for i, (is_valid, values) in enumerate(rows):
        if is_valid:
            out[i] = torch.from_numpy(values) / sigma
    return out


def _rdkit_descriptor_row(smi: object) -> dict[str, str | bool | float]:
    if pd.isna(smi) or not isinstance(smi, str) or smi == "":
        mol = None
        smi_str = ""
    else:
        smi_str = smi
        mol = Chem.MolFromSmiles(smi_str)

    if mol is None:
        return {
            "smiles": smi_str,
            "valid": False,
            "MolWt": np.nan,
            "MolLogP": np.nan,
            "TPSA": np.nan,
            "NumRotatableBonds": np.nan,
        }

    return {
        "smiles": smi_str,
        "valid": True,
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
    }


def rdkit_descriptor_df(smiles_list: list[str | None]) -> pd.DataFrame:
    """
    Compute a small, interpretable RDKit descriptor panel for each SMILES.

    Invalid SMILES are retained in the output with valid=False and NaN
    descriptor values so callers can decide whether to drop them.
    """
    rows = [_rdkit_descriptor_row(smi) for smi in smiles_list]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pre-compute and cache target features from MOSES splits
# ---------------------------------------------------------------------------


def load_moses_split_smiles(
    cache_dir: str | Path,
    split: str,
) -> list[str]:
    """Load raw SMILES for one MOSES split, downloading the CSV if needed."""
    import pandas as pd
    from torch_geometric.data import download_url

    if split not in _MOSES_SPLIT_URLS:
        raise ValueError(f"Unknown MOSES split: {split}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{split}_moses.csv"
    if not csv_path.exists():
        print(f"Downloading MOSES {split} CSV...")
        downloaded = Path(download_url(_MOSES_SPLIT_URLS[split], str(cache_dir)))
        downloaded.rename(csv_path)

    return pd.read_csv(csv_path)["SMILES"].tolist()


def precompute_moses_features(
    cache_dir: str | Path,
    split: str,
    n_bits: int = 256,
    fcd_device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Download one MOSES split CSV (if not cached), compute Morgan FP and ChemNet
    features for all molecules, and save to cache_dir/{split}_features.pt.
    On subsequent calls the cached file is loaded directly.

    Bypasses MOSESDataset graph processing entirely — only the raw CSV is needed.

    Args:
        cache_dir:    Root directory for data and feature cache.
        split:       One of {'train', 'val', 'test'}.
        n_bits:      Morgan fingerprint length.
        fcd_device:  Device for ChemNet model ('cpu' or 'cuda').

    Returns:
        dict with keys 'morgan' ([M, n_bits]) and 'fcd' ([M, 512]).
    """
    if split not in _MOSES_SPLIT_URLS:
        raise ValueError(f"Unknown MOSES split: {split}")

    cache_dir = Path(cache_dir)
    save_path = cache_dir / f"{split}_features.pt"

    if save_path.exists():
        print(f"Loading cached {split} features from {save_path}")
        return torch.load(save_path, weights_only=True)

    smiles_list = load_moses_split_smiles(cache_dir, split)
    print(f"Computing features for {len(smiles_list)} {split} molecules...")

    morgan = morgan_features(smiles_list, n_bits=n_bits)
    print(f"  Morgan FP done: {morgan.shape}")

    fcd_model = load_fcd_model(device=fcd_device)
    fcd = fcd_features(smiles_list, fcd_model=fcd_model)
    print(f"  ChemNet FCD done: {fcd.shape}")

    result = {"morgan": morgan, "fcd": fcd}
    torch.save(result, save_path)
    print(f"Saved to {save_path}")
    return result


def precompute_val_features(
    cache_dir: str | Path,
    n_bits: int = 256,
    fcd_device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Compatibility wrapper for callers that still want the val split."""
    return precompute_moses_features(
        cache_dir,
        split="val",
        n_bits=n_bits,
        fcd_device=fcd_device,
    )


# ---------------------------------------------------------------------------
# Pre-compute and cache target features from GuacaMol splits
# ---------------------------------------------------------------------------

_GUACAMOL_SPLIT_URLS = {
    "train": "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/13612760/guacamol_v1_train.smiles",
    "val": "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/13612766/guacamol_v1_valid.smiles",
    "test": "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/13612757/guacamol_v1_test.smiles",
}


def load_guacamol_split_smiles(
    cache_dir: str | Path,
    split: str,
) -> list[str]:
    """Load raw SMILES for one GuacaMol split, downloading the .smiles file if needed."""
    import urllib.request

    if split not in _GUACAMOL_SPLIT_URLS:
        raise ValueError(f"Unknown GuacaMol split: {split}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_path = cache_dir / f"guacamol_v1_{split}.smiles"
    if not smiles_path.exists():
        print(f"Downloading GuacaMol {split} SMILES...")
        urllib.request.urlretrieve(_GUACAMOL_SPLIT_URLS[split], smiles_path)

    with open(smiles_path) as f:
        return [line.strip() for line in f if line.strip()]


def precompute_guacamol_features(
    cache_dir: str | Path,
    split: str = "val",
    n_bits: int = 256,
    fcd_device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Download one GuacaMol split (if not cached), compute Morgan FP and ChemNet
    features, and save to cache_dir/guacamol_{split}_features.pt.

    Returns:
        dict with keys 'morgan' ([M, n_bits]) and 'fcd' ([M, 512]).
    """
    if split not in _GUACAMOL_SPLIT_URLS:
        raise ValueError(f"Unknown GuacaMol split: {split}")

    cache_dir = Path(cache_dir)
    save_path = cache_dir / f"guacamol_{split}_features.pt"

    if save_path.exists():
        print(f"Loading cached GuacaMol {split} features from {save_path}")
        return torch.load(save_path, weights_only=True)

    smiles_list = load_guacamol_split_smiles(cache_dir, split)
    print(f"Computing features for {len(smiles_list)} GuacaMol {split} molecules...")

    morgan = morgan_features(smiles_list, n_bits=n_bits)
    print(f"  Morgan FP done: {morgan.shape}")

    fcd_model = load_fcd_model(device=fcd_device)
    fcd = fcd_features(smiles_list, fcd_model=fcd_model)
    print(f"  ChemNet FCD done: {fcd.shape}")

    result = {"morgan": morgan, "fcd": fcd}
    torch.save(result, save_path)
    print(f"Saved to {save_path}")
    return result
