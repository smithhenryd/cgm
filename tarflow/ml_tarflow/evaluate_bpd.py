import argparse
import os
import pathlib

import numpy as np
import torch
import torch.utils.data
import torchvision as tv
import transformer_flow


def gaussian_log_prob(z: torch.Tensor, k: int = 128) -> torch.Tensor:
    log_p = -0.5 * (z**2 + np.log(2 * np.pi))
    log_p = log_p.flatten(1).sum(-1) - np.log(k) * np.prod(z.size()[1:])
    return log_p


def main(args):
    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(args.img_size),
            tv.transforms.CenterCrop(args.img_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data = tv.datasets.ImageFolder(
        os.path.join(args.data, args.dataset, "val"),
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda")
    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=args.num_classes,
    ).to(device)

    ckpt_file = args.ckpt_file
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    print("Starting BPD evaluation")

    all_bpd = 0
    cnt = 0
    n_dims = args.img_size * args.img_size * 3

    for x, y in data_loader:
        x = x.to(device)
        x_int = (x + 1) * (255 / 2)
        x = (x_int + torch.rand_like(x_int)) / 256
        x = x * 2 - 1
        y = None
        with torch.no_grad():
            z, outputs, logdets = model(x, y)
            prior_log_p = gaussian_log_prob(z)
            nll = -prior_log_p / n_dims - logdets
            bpd = nll / np.log(2)
            batch_bpd = bpd.sum().item()
            all_bpd += batch_bpd
            cnt += z.size(0)
            running_mean_bpd = all_bpd / cnt
            print(f"Running mean BPD: {running_mean_bpd:.4f}")

    final_bpd = all_bpd / cnt
    print(f"BPD: {final_bpd:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", type=pathlib.Path)
    parser.add_argument("--ckpt_file", default="", type=str)
    parser.add_argument(
        "--dataset",
        default="imagenet64",
        type=str,
        choices=["imagenet", "imagenet64", "afhq"],
    )
    parser.add_argument("--img_size", default=64, type=int)
    parser.add_argument("--num_classes", default=0, type=int)
    parser.add_argument("--channel_size", default=3, type=int)

    parser.add_argument("--patch_size", default=2, type=int)
    parser.add_argument("--channels", default=768, type=int)
    parser.add_argument("--blocks", default=8, type=int)
    parser.add_argument("--layers_per_block", default=8, type=int)
    parser.add_argument("--nvp", default=1, type=int)
    parser.add_argument("--batch_size", default=500, type=int)

    args = parser.parse_args()

    main(args)
