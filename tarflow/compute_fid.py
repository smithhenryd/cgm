import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torch
import torchvision as tv
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.image.fid import FrechetInceptionDistance

from PIL import Image, ImageOps
from tqdm import tqdm

from normalizing_flow import NormalizingFlow
from tarflow_map import TarflowMap

# ------------------ Preprocessing for AFHQ images ------------------
def ensure_rgb_pil(path: str) -> Image.Image:
    """
    Ensures image is three-channel RBG
    """
    img = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(img)


def save_png_from_tensor01(x_chw: torch.Tensor, dst_path: str):
    """
    Saves a tensor with values in [0, 1] to png
    """
    x = x_chw.detach().float().clamp(0, 1)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tv.utils.save_image(x, dst_path, normalize=False) # do NOT normalize the image


def denorm_minus1_1_to_0_1(x: torch.Tensor) -> torch.Tensor:
    """
    Transform a tensor with values [-1,1] -> [0,1]
    """
    return 0.5 * (img.clamp(-1, 1) + 1.)


def preprocess_afhq_eval(
    afhq_root: str,
    split: str,
    out_dir: Union[str, Path],
    img_size: int = 256,
):
    print(f"[AFHQ] Preprocessing {split} -> {img_size}x{img_size} into {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(img_size),
            tv.transforms.CenterCrop(img_size),
        ]
    )

    src_root = os.path.join(afhq_root, split)
    classes = ["wild"]  # use only wildlife samples

    for cls in classes:
        cls_dir = os.path.join(src_root, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Missing directory: {cls_dir}")

        for root, _, files in os.walk(cls_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in {".jpg", ".jpeg", ".png"}:
                    continue

                src_path = os.path.join(root, fname)
                rel = os.path.relpath(src_path, start=src_root)
                dst_path = os.path.join(out_dir, os.path.splitext(rel)[0] + ".png")
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                img = ensure_rgb_pil(src_path)
                img = transform(img)
                img.save(dst_path, format="PNG", compress_level=3)


# ------------------ Sampling from TarFlow model ------------------
def load_state_dict(model: NormalizingFlow, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model_state" in ckpt:
        sd = ckpt["model_state"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)


def generate_samples_to_dir(
    flow: NormalizingFlow,
    out_dir: str,
    n_samples: int,
    batch_size: int = 128,
    img_size: int = 256,
    channel_size: int = 3,
    device: str = "cuda",
):
    """
    Save n_samples PNGs to out_dir as img_size x img_size in [0,1]
    """

    class_name = "wild"
    class_dir = os.path.join(out_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    flow.eval()
    with torch.no_grad():
        idx = 0
        pbar = tqdm(total=n_samples, desc="Generating samples", unit="img")
        while idx < n_samples:
            b = min(batch_size, n_samples - idx)
            samps = flow.sample(b)
            xs = samps.xs.view(b, channel_size, img_size, img_size)
            xs = denorm_minus1_1_to_0_1(xs)  # convert image from [-1, 1] -> [0, 1]
            for i in range(b):
                out_path = os.path.join(out_dir, f"sample_{idx + i:06d}.png")
                save_png_from_tensor01(xs[i], out_path)
            idx += b
            pbar.update(b)
        pbar.close()


# ------------------ Compute FID ------------------
def make_fid_loader(path: str, batch_size: int, num_workers: int):
    """
    Instantiates a DataLoader object for computing FID
    """
    tfm = tv.transforms.Compose(
        [
            tv.transforms.Resize(
                299, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            tv.transforms.CenterCrop(299),
            tv.transforms.ToTensor(),  # [0,1],
        ]
    )
    ds = ImageFolder(path, transform=tfm)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )


class _PathListDataset(Dataset):
    def __init__(self, paths, labels, transform=None, root=None):
        self.root = root
        self.paths = paths
        self.labels = labels 
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        p = p if os.path.isabs(p) or self.root is None else os.path.join(self.root, p)
        with Image.open(p) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, self.labels[i]


def _to_class_id(one_hot_or_id):
    if isinstance(one_hot_or_id, (int,)):
        return int(one_hot_or_id)
    if torch.is_tensor(one_hot_or_id):
        return int(torch.argmax(one_hot_or_id).item())
    return int(torch.tensor(one_hot_or_id).argmax().item())


def make_weighted_fid_loader(path: str, label_dict: dict, batch_size: int, num_workers: int, target_prior=None, num_samples=None, alpha: float = 1.0, seed: int = 0):
  
    filepaths, y = [], []
    for fp, enc in label_dict.items():
        full = fp if os.path.isabs(fp) else os.path.join(path, fp)
        if os.path.isfile(full):
            filepaths.append(fp)           
            y.append(_to_class_id(enc))
    if not filepaths:
        raise ValueError("No valid image paths found in label_dict for the given root.")

    y = torch.as_tensor(y, dtype=torch.long)
    K = int(y.max().item()) + 1

    # Transforms
    tfm = tv.transforms.Compose(
        [
            tv.transforms.Resize(299, interpolation=InterpolationMode.BICUBIC, antialias=True),
            tv.transforms.CenterCrop(299),
            tv.transforms.ToTensor(),  # [0,1]
        ]
    )

    ds = _PathListDataset(filepaths, y.tolist(), transform=tfm, root=path)

    counts = torch.bincount(y, minlength=K).float()
    p_hat = counts / counts.sum()
    if target_prior is None:
        q = torch.full_like(p_hat, 1.0 / K)
    else:
        q = torch.as_tensor(target_prior, dtype=torch.float)[:K]
        q = q / q.sum()

    # Per-sample weights and sampler (WITH replacement)
    class_w = (q / p_hat.clamp_min(1e-12)).pow(alpha)
    sample_w = class_w[y]

    if num_samples is None:
        num_samples = len(ds)

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_w,
        num_samples=int(num_samples),
        replacement=True,
        generator=g,
    )

    # DataLoader
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def compute_fid(
    real_dir: str, fake_dir: str, weight_dict: Union[Path, str], batch_size: int, num_workers: int, device: str
):
    """
    Computes FID from two image directories: real_dir, containing images from the training dataset and fake_dir, containing images generated by the model
    """
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048).to(device_t)

    if weight_dict is not None:
        label_dict = torch.load(weight_dict, map_location=device_t)
        real_loader = make_weighted_fid_loader(real_dir, label_dict, batch_size, num_workers)
    else:
        real_loader = make_fid_loader(real_dir, batch_size, num_workers)
    fake_loader = make_fid_loader(fake_dir, batch_size, num_workers)

    with torch.no_grad():
        for x, _ in tqdm(real_loader, desc="FID real", leave=False):
            x = (x * 255).to(torch.uint8).to(device_t)
            fid.update(x, real=True)
        for x, _ in tqdm(fake_loader, desc="FID fake", leave=False):
            x = (x * 255).to(torch.uint8).to(device_t)
            fid.update(x, real=False)

    return float(fid.compute())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afhq_root", type=str) # AFHQ root dir containing train/ and val/
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val"]
    )
    parser.add_argument("--out_fake_dir", type=str) # Where to write/generated samples for FID
    parser.add_argument("--path", type=str, default=None) # Path to base TarFlow model
    parser.add_argument("--ckpt", type=str, default=None) # Path to TarFlow checkpoint (fine-tuned)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--weight_dict", type=str, default=None)
    args = parser.parse_args()

    (
        channel_size,
        img_size,
        patch_size,
        channels,
        num_blocks,
        layers_per_block,
        num_classes,
        noise_std,
    ) = (3, 256, 8, 768, 8, 8, 3, 0.07)
    dim = (img_size // patch_size) ** 2 * channel_size * (patch_size**2)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Preprocess AFHQ deterministically
    real_outdir = os.path.join(args.afhq_root, f"tmp_{args.split}_{img_size}")
    if os.path.isdir(real_outdir) and any(os.scandir(real_outdir)):
        print(
            f"[INFO] Samples already exist in {real_outdir}; skipping preprocessing"
        )
    else:
        preprocess_afhq_eval(args.afhq_root, args.split, real_outdir, img_size=img_size)

    # Build flow model, load weights, and generate samples if they don't already exist
    # Skip generation if directory exists and is non-empty
    if os.path.isdir(args.out_fake_dir) and any(os.scandir(args.out_fake_dir)):
        print(
            f"[INFO] Samples already exist in {args.out_fake_dir}; skipping generation"
        )
    else:
        print(
            f"[INFO] Generating {args.n_samples} fake samples into {args.out_fake_dir}"
        )
        # Load checkpoint
        path = args.path
        tarflow_map = TarflowMap(
            path,
            device,
            channel_size,
            img_size,
            patch_size,
            channels,
            num_blocks,
            layers_per_block,
            num_classes,
            noise_std,
            sample_cls=2,
            sampling_args={"guidance": 1.0, "attn_temp": 1.0},
        )
        model = NormalizingFlow(dim=dim, map=tarflow_map)
        load_state_dict(flow, args.ckpt, device=device)

        # Generate samples
        generate_samples_to_dir(
            model,
            args.out_fake_dir,
            args.n_samples,
            args.batch_size,
            img_size,
            channel_size,
            args.device,
        )

    # Compute FID
    fid = compute_fid(
        real_outdir,
        args.out_fake_dir,
        weight_dict=args.weight_dict,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"FID ({args.split} @ {img_size}): {fid:.4f}")

if __name__ == "__main__":
    main()