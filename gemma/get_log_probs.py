import os
from argparse import ArgumentParser
from pathlib import Path

import torch

from cgm.cgm import log_p_chunked
from cgm_gemma import Sample, ProfessionStories

Sample.__module__ = "__main__"  # necessary for weight_only load
torch.serialization.add_safe_globals([Sample])


def default_output_path(
    *, output_folder: str, sample_path: str, ft_model_path: str
) -> str:
    sample_stem = Path(sample_path).stem
    ft_stem = Path(ft_model_path).stem
    fname = f"{sample_stem}__ft_{ft_stem}__logps.pt"
    return os.path.join(output_folder, fname)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=64)
    args = parser.parse_args()

    sample_path: str = args.sample_path
    ft_model_path: str = args.ft_model_path
    output_folder: str = args.output_folder
    output_path: str = args.output_path or default_output_path(
        output_folder=output_folder,
        sample_path=sample_path,
        ft_model_path=ft_model_path,
    )
    chunk_size: int = args.chunk_size

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Bad sample_path: {sample_path}")
    if not os.path.exists(ft_model_path):
        raise FileNotFoundError(f"Bad ft_model_path: {ft_model_path}")

    print(f"Loading data from {sample_path}")
    data = torch.load(sample_path, weights_only=True)
    sample = data["post_samples"]

    model = ProfessionStories()

    N = len(sample)
    if N % chunk_size != 0:
        raise ValueError(
            f"N must be divisible by chunk_size; got N={N}, chunk_size={chunk_size}",
        )
    batch_chunks = N // chunk_size

    print("Base model log probs")
    with torch.no_grad():
        base_log_p = (
            log_p_chunked(model, sample, batch_size=N, batch_chunks=batch_chunks)
            .cpu()
            .float()
        )

    print(f"Loading finetuned model from {ft_model_path}")
    state_dict = torch.load(ft_model_path, weights_only=True)
    model.load_state_dict(state_dict)

    print("FT model log probs")
    with torch.no_grad():
        ft_log_p = (
            log_p_chunked(model, sample, batch_size=N, batch_chunks=batch_chunks)
            .cpu()
            .float()
        )

    data["base_log_p"] = base_log_p
    data["ft_log_p"] = ft_log_p

    # Optional provenance to make downstream debugging easier
    data["sample_path"] = sample_path
    data["ft_model_path"] = ft_model_path
    data["chunk_size"] = chunk_size

    print(f"Saving results to {output_path}")
    torch.save(data, output_path)
