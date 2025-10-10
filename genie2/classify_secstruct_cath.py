"""
Compute Biotite P-SEA secondary structure fractions (helix/strand) for CATH domains in Ingraham et al. (2019) chain_set.jsonl

Usage:
  python classify_sectruct_cath.py \
    --jsonl https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl \
    -o cath_biotite_ss.csv
"""

import argparse
import io
import json
import math
import sys
import re
from typing import Iterable, Dict, Any, List, Optional
import numpy as np

# If streaming from a URL
try:
    import requests
except Exception:
    requests = None

# For secondary structure annotation
import biotite.structure as struc
from biotite.structure import annotate_sse


def iter_jsonl(path_or_url: str):
    """
    Stream JSONL either from a local file or via HTTP(S)
    """
    if path_or_url.startswith(("http://", "https://")):
        if requests is None:
            raise RuntimeError("pip install requests to read HTTP(S) URLs")
        r = requests.get(path_or_url, stream=True, timeout=60)
        r.raise_for_status()
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                yield json.loads(line)
        finally:
            r.close()
    else:
        with open(path_or_url, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def to_ca_coords(coords_obj) -> np.ndarray:
    """
    coords_obj is a dict with keys like 'N','CA','C','O' mapping to (L,3)
    Extract only the CA array
    """
    try:
        ca = np.asarray(coords_obj["CA"], dtype=float)
    except Exception as e:
        ks = (
            list(coords_obj.keys())[:6]
            if isinstance(coords_obj, dict)
            else type(coords_obj)
        )
        raise ValueError(f"Expected coords['CA'] -> (L,3); got keys={ks}") from e
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError(f"'CA' has shape {ca.shape}, expected (L,3)")
    mask = np.isfinite(ca).all(axis=1)
    ca = ca[mask]
    if ca.size == 0:
        raise ValueError("No finite CA rows after masking.")
    return ca.astype(np.float32, copy=False)


def build_ca_atomarray(ca: np.ndarray) -> struc.AtomArray:
    """
    Builds an AtomArray object from CA coordinates
    """
    L = ca.shape[0]
    arr = struc.AtomArray(L)
    arr.chain_id[:] = "A"
    arr.res_id = np.arange(1, L + 1, dtype=int)
    arr.ins_code[:] = ""
    arr.res_name[:] = "GLY"  # arbitrary std residue; P-SEA is CA-only
    arr.hetero[:] = False
    arr.atom_name[:] = "CA"
    arr.element[:] = "C"
    arr.coord = ca
    return arr


def frac_helix_strand(atom_array: struc.AtomArray) -> tuple[float, float]:
    """
    Annotates secondary structure in an AtomArray according to biotite annotate_sse
    """
    sse = annotate_sse(atom_array)  # 'a','b','c',''
    L = len(sse)
    if L == 0:
        return (math.nan, math.nan)
    return (
        float(np.count_nonzero(sse == "a") / L),
        float(np.count_nonzero(sse == "b") / L),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("-o", "--output", default="cath_biotite_ss.csv")
    args = ap.parse_args()

    out = open(args.output, "w", encoding="utf-8")
    print("pdb_path,% Helix,% Strand", file=out)

    it = iter_jsonl(args.jsonl)
    n_total = n_ok = n_err = 0
    for rec in it:
        n_total += 1
        try:
            name = rec["name"]
            ca = to_ca_coords(rec["coords"])
            if ca.shape[0] < 2:
                # Write zeros if too short to classify sensibly
                print(f"{name},0.0000000000000000,0.0000000000000000", file=out)
                n_ok += 1
                continue
            arr = build_ca_atomarray(ca)
            h, b = frac_helix_strand(arr)
            print(f"{name},{h:.16f},{b:.16f}", file=out)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(
                f"[warn] skipping '{rec.get('name','?')}' at idx {n_total}: {e}",
                file=sys.stderr,
            )
    out.close()
    print(
        f"Finished SS annotation: Wrote {n_ok} rows to {args.output} (total={n_total}, errors={n_err})"
    )


if __name__ == "__main__":
    main()