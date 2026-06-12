import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))


DEFAULT_DATA_PATH = Path(
    "/scratch/users/diamant/data/the_code/General/data/DeepMEL2_data.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute AlphaGenome peak ATAC features for DeepMEL2 sequences "
            "and save a split-separated target cache."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="DeepMEL2 pickle with train/valid/test one-hot DNA and conditions.",
    )
    parser.add_argument(
        "--output-pt",
        type=Path,
        required=True,
        help="Output .pt file for AlphaGenome features and metadata.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "val", "test"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--max-num-per-condition",
        type=int,
        default=None,
        help=(
            "Maximum number of positive examples to select per condition and split. "
            "When omitted, all positive examples are selected."
        ),
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=2,
        help="AlphaGenome forward-pass batch size.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional AlphaGenome model path override.",
    )
    parser.add_argument(
        "--background-file",
        type=str,
        default=None,
        help="Optional background .npy file used by enhancers/features.py.",
    )
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
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_key(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys were found in the data pickle: {keys}")


def split_keys(split: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if split == "train":
        return ("train_data",), ("y_train", "train_y")
    if split in {"valid", "val"}:
        return ("valid_data", "val_data"), ("y_valid", "valid_y", "y_val", "val_y")
    if split == "test":
        return ("test_data",), ("y_test", "test_y")
    raise ValueError(f"Unknown split {split!r}")


def one_hot_to_tokens(dna: Any) -> torch.Tensor:
    dna = torch.as_tensor(dna)
    if dna.ndim != 3 or dna.shape[-1] != 4:
        raise ValueError(
            f"Expected one-hot DNA with shape (n, length, 4), got {dna.shape}"
        )
    return dna.argmax(dim=-1).long()


def condition_tensor(cond: Any) -> torch.Tensor:
    cond = torch.as_tensor(cond)
    if cond.ndim != 2:
        raise ValueError(
            f"Expected condition matrix with shape (n, cond_dim), got {cond.shape}"
        )
    return cond.float()


def load_split(data: dict[str, Any], split: str) -> tuple[torch.Tensor, torch.Tensor]:
    dna_keys, cond_keys = split_keys(split)
    return (
        one_hot_to_tokens(get_key(data, *dna_keys)),
        condition_tensor(get_key(data, *cond_keys)),
    )


def condition_key(condition_idx: int) -> str:
    return f"condition_{condition_idx}"


def select_condition_examples(
    cond: torch.Tensor,
    max_num_per_condition: int | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    condition_to_sequence_indices: dict[str, torch.Tensor] = {}
    selected_chunks: list[torch.Tensor] = []
    positive = cond > 0

    for condition_idx in range(cond.shape[1]):
        idx = torch.nonzero(positive[:, condition_idx], as_tuple=False).squeeze(1)
        if max_num_per_condition is not None and idx.numel() > max_num_per_condition:
            perm = torch.randperm(idx.numel(), generator=generator)
            idx = idx[perm[:max_num_per_condition]]
        idx = idx.sort().values
        key = condition_key(condition_idx)
        condition_to_sequence_indices[key] = idx
        if idx.numel() > 0:
            selected_chunks.append(idx)

    if not selected_chunks:
        raise ValueError("No positive condition examples were found.")

    selected_sequence_indices = torch.cat(selected_chunks).unique(sorted=True)
    feature_row_for_sequence = torch.full((cond.shape[0],), -1, dtype=torch.long)
    feature_row_for_sequence[selected_sequence_indices] = torch.arange(
        selected_sequence_indices.numel()
    )

    condition_to_indices = {}
    for key, sequence_indices in condition_to_sequence_indices.items():
        feature_indices = feature_row_for_sequence[sequence_indices]
        if (feature_indices < 0).any():
            raise RuntimeError(f"Internal selection error for {key}")
        condition_to_indices[key] = feature_indices

    return (
        selected_sequence_indices,
        condition_to_indices,
        condition_to_sequence_indices,
    )


def compute_split_features(
    split: str,
    tokens: torch.Tensor,
    cond: torch.Tensor,
    feature_extractor,
    max_num_per_condition: int | None,
    generator: torch.Generator,
) -> dict[str, Any]:
    selected_indices, condition_to_indices, condition_to_sequence_indices = (
        select_condition_examples(cond, max_num_per_condition, generator)
    )
    selected_tokens = tokens[selected_indices]
    selected_cond = cond[selected_indices]

    features = feature_extractor(selected_tokens).detach().cpu().to(torch.float32)

    selected_condition_counts = torch.tensor(
        [condition_to_indices[condition_key(i)].numel() for i in range(cond.shape[1])],
        dtype=torch.long,
    )

    return {
        "features": features,
        "sequence_tokens": selected_tokens.cpu(),
        "condition_matrix": selected_cond.cpu(),
        "sequence_indices": selected_indices.cpu(),
        "condition_to_indices": {
            key: value.cpu() for key, value in condition_to_indices.items()
        },
        "condition_to_sequence_indices": {
            key: value.cpu() for key, value in condition_to_sequence_indices.items()
        },
        "condition_keys": [condition_key(i) for i in range(cond.shape[1])],
        "full_condition_counts": (cond > 0).sum(dim=0).cpu().to(torch.long),
        "selected_condition_counts": selected_condition_counts,
        "num_total_sequences": int(tokens.shape[0]),
        "num_selected_sequences": int(selected_tokens.shape[0]),
        "sequence_length": int(tokens.shape[1]),
        "condition_dim": int(cond.shape[1]),
        "feature_dim": int(features.shape[1]),
        "split": split,
    }


def build_feature_extractor_from_args(args: argparse.Namespace):
    from features import build_feature_extractor

    kwargs = {
        "batch_size": args.feature_batch_size,
        "autocast_dtype": args.alphagenome_autocast_dtype,
        "matmul_precision": args.matmul_precision,
        "verbose": args.verbose,
    }
    if args.model_path is not None:
        kwargs["model_path"] = args.model_path
    if args.background_file is not None:
        kwargs["bg_file"] = args.background_file
    return build_feature_extractor(**kwargs)


def main() -> None:
    args = parse_args()
    if args.max_num_per_condition is not None and args.max_num_per_condition < 1:
        raise ValueError("--max-num-per-condition must be positive when provided.")

    with args.data_path.open("rb") as f:
        all_data = pickle.load(f)

    feature_extractor = build_feature_extractor_from_args(args)
    generator = torch.Generator().manual_seed(args.seed)
    splits: dict[str, dict[str, Any]] = {}

    for split in tqdm(args.splits, desc="Processing splits"):
        output_split = "valid" if split == "val" else split
        tokens, cond = load_split(all_data, split)
        splits[output_split] = compute_split_features(
            split=output_split,
            tokens=tokens,
            cond=cond,
            feature_extractor=feature_extractor,
            max_num_per_condition=args.max_num_per_condition,
            generator=generator,
        )

    payload = {
        "splits": splits,
        "data_path": str(args.data_path),
        "max_num_per_condition": args.max_num_per_condition,
        "feature_batch_size": args.feature_batch_size,
        "alphagenome_autocast_dtype": args.alphagenome_autocast_dtype,
        "matmul_precision": args.matmul_precision,
        "model_path": args.model_path,
        "background_file": args.background_file,
        "seed": args.seed,
        "condition_key_format": "condition_{condition_idx}",
    }

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_pt)
    print(f"Saved AlphaGenome feature cache to {args.output_pt}")
    for split, split_payload in splits.items():
        print(
            f"{split}: features={tuple(split_payload['features'].shape)}, "
            f"selected_sequences={split_payload['num_selected_sequences']}"
        )


if __name__ == "__main__":
    main()
