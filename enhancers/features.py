from functools import partial

from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

DEFAULT_MODEL_PATH = (
    "/scratch/users/diamant/models/alphagenome/model_all_folds.safetensors"
)
DEFAULT_BG = "background.npy"
AUTOCAST_DTYPES = {
    "none": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def set_matmul_precision(matmul_precision: str | None) -> None:
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    from alphagenome_pytorch import AlphaGenome

    model = AlphaGenome.from_pretrained(model_path, device="cuda").eval()
    model.predict = torch.compile(model.predict, mode="reduce-overhead")
    return model


def load_background(bg_file: str = DEFAULT_BG) -> torch.Tensor:
    return torch.tensor(np.load(bg_file), device="cuda")


def insert_seqs(seqs: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """
    seqs: M x seq len (usually 500)
    background: background len (usually 2048)
    """
    seqs = seqs.to(device=background.device, dtype=background.dtype)
    M, L = seqs.shape
    start = (background.shape[0] - L) // 2
    inserted = background.unsqueeze(0).repeat(M, 1)
    inserted[:, start : start + L] = seqs
    return inserted


def onehot_for_model(x: torch.Tensor) -> torch.Tensor:
    return F.one_hot(torch.as_tensor(x, dtype=torch.long, device=x.device), 4)


@torch.no_grad()
def batched_features(
    seqs: torch.Tensor,
    background: torch.Tensor,
    model,
    batch_size: int = 2,
    autocast_dtype: str = "none",
    verbose: bool = False,
) -> torch.Tensor:
    """
    seqs: bsz x seq_len
    background: background len
    """
    dtype = AUTOCAST_DTYPES[autocast_dtype]
    ins = insert_seqs(seqs, background)
    x = onehot_for_model(ins)

    M = ins.shape[0]
    features = []
    for start in tqdm(
        range(0, M, batch_size),
        desc="Getting AlphaGenome predictions",
        disable=not verbose,
    ):
        stop = start + batch_size
        with torch.autocast(
            device_type=x.device.type,
            dtype=dtype or torch.bfloat16,
            enabled=dtype is not None,
        ):
            pred = model.predict(x[start:stop], organism_index=0)
        # last 89 atac channels are all zeros
        atac = pred["atac"][1][..., :-89]  # bsz x bg len x num. ATAC
        peaks = atac.max(1).values  # bsz x num. ATAC
        features.append(peaks)

    return torch.cat(features, dim=0)


def build_feature_extractor(
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = 2,
    bg_file: str = DEFAULT_BG,
    autocast_dtype: str = "none",
    matmul_precision: str | None = None,
    verbose: bool = False,
):
    if autocast_dtype not in AUTOCAST_DTYPES:
        raise ValueError(
            f"autocast_dtype must be one of {sorted(AUTOCAST_DTYPES)}, "
            f"got {autocast_dtype!r}"
        )
    set_matmul_precision(matmul_precision)
    model = load_model(model_path)
    background = load_background(bg_file)
    return partial(
        batched_features,
        background=background,
        model=model,
        batch_size=batch_size,
        autocast_dtype=autocast_dtype,
        verbose=verbose,
    )
