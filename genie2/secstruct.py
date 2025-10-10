from biotite.structure import annotate_sse
import biotite.structure.io.pdb as pdb

from genie.utils.feat_utils import save_np_features_to_pdb

import numpy as np

import random
import string
import sys

import re
import os
import glob


def fix_pdb_columns(pdb_path: str) -> str:
    """
    Reads a PDB file, fixes missing occupancy and B-factor columns, and writes a corrected file with '_fixed' appended to the filename
    """
    fixed_pdb_path = pdb_path.replace(".pdb", "_fixed.pdb")

    with open(pdb_path, "r") as infile, open(fixed_pdb_path, "w") as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                # Ensure correct formatting using PDB column specifications
                # Occupancy (54-59), B-factor (60-65)
                fixed_line = line[:54].ljust(54) + " 1.00  0.00" + line[66:]
                outfile.write(fixed_line + "\n")
            else:
                outfile.write(line)  # Write other lines unchanged
    return fixed_pdb_path


def sec_struct_frac(pdb_path: str) -> np.ndarray:
    """
    Returns the fraction of residues of the pdb in [helix, strand, coil]
    """

    fixed_pdb_path = fix_pdb_columns(pdb_path)
    pdb_file = pdb.PDBFile.read(fixed_pdb_path)

    # Convert to AtomArray (or AtomArrayStack for multiple models)
    atom_array = pdb.get_structure(pdb_file)

    # Remove fixed pdb file
    os.remove(fixed_pdb_path)

    sse = annotate_sse(atom_array[0])
    return np.array(
        [sum([ss_val == ss_type for ss_val in sse]) for ss_type in ["a", "b", "c"]]
    ) / len(sse)


def save_pdb(x: np.ndarray, pdb_path: str) -> None:
    """
    Writes a Ca only pdb file using Genie2 utility function.

    x: a shape (N, 3) array representing Ca coordinates, units of Angstroms
    """
    assert x.ndim == 2 and x.shape[1] == 3
    N = x.shape[0]
    np_features = {
        "atom_positions": x,
        "aatype": np.array([np.eye(N)[0] for _ in range(N)]),
        "residue_index": np.arange(N),
        "chain_index": np.zeros(N, dtype=int),
        "fixed_group": np.zeros(N, dtype=int),
    }
    save_np_features_to_pdb(np_features, pdb_path)


def compute_quantiles(x: np.ndarray, quantile_vals: dict[str:float]) -> dict:
    """
    Determines whether or not the proportion of residues in x belonging to SS [alpha, beta] is at
    most equal to the specified quantile values

    x: a shape (N, 3) array representing Ca coordinates, units of Angstroms
    quantile_vals: a dictionary with keys 'alpha' and 'beta'; each item is a list containing the SS quantiles
    """

    tmp = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    pdb_path = tmp + ".pdb"
    save_pdb(x, pdb_path)
    try:
        ss_fracs = sec_struct_frac(pdb_path)
    finally:
        os.remove(pdb_path)

    key_to_idx = {"alpha": 0, "beta": 1}
    out = {}
    for key, thresh_list in quantile_vals.items():
        if key not in key_to_idx:
            raise KeyError(f"Unknown key {key!r}; expected 'alpha' or 'beta'")
        frac = ss_fracs[key_to_idx[key]]
        out[key] = [frac <= q for q in thresh_list]
    return out


def compute_joint_quantiles(x: np.ndarray, quantile_vals: dict[str:list]):
    """
    Determines whether or not the proportion of residues in x belonging to SS [alpha, beta] is at
    most equal to the specified quantile values, jointly for alpha and beta

    x: a shape (N, 3) array representing Ca coordinates, units of Angstroms
    quantile_vals: a dictionary with keys 'alpha' and 'beta'; each item is a list containing the joint
    SS quantiles, both lists must have the same length
    """

    tmp = "".join(random.choices(string.ascii_letters + string.digits, k=12))
    pdb_path = tmp + ".pdb"
    save_pdb(x, pdb_path)

    try:
        ss_fracs = sec_struct_frac(pdb_path)
    finally:
        os.remove(pdb_path)

    frac_alpha, frac_beta = ss_fracs[0], ss_fracs[1]

    a_list = quantile_vals.get("alpha", [])
    b_list = quantile_vals.get("beta", [])
    assert len(a_list) == len(b_list)

    return [
        (frac_alpha <= a_thresh) and (frac_beta <= b_thresh)
        for a_thresh, b_thresh in zip(a_list, b_list)
    ]
