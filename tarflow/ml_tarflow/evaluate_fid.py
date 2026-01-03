#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import builtins
import pathlib

import numpy as np
import torch
import torch.utils.data
import torchvision as tv

import transformer_flow
import utils


def main(args):
    args.denoising_batch_size = args.batch_size // 4
    dist = utils.Distributed()
    utils.set_random_seed(100 + dist.rank)
    num_classes = utils.get_num_classes(args.dataset)

    def print(*args, **kwargs):
        if dist.local_rank == 0:
            builtins.print(*args, **kwargs)

    # check if the fid stats had been previously computed
    fid_stats_file = f"{args.dataset}_{args.img_size}_fid_stats.pth"
    fid_stats_file = args.data / f"{args.dataset}_{args.img_size}_fid_stats.pth"
    assert fid_stats_file.exists()
    print(f"Loading FID stats from {fid_stats_file}")
    fid = utils.FID(reset_real_features=False, normalize=True).cuda()
    fid.load_state_dict(
        torch.load(fid_stats_file, map_location="cpu", weights_only=False)
    )
    dist.barrier()

    model = transformer_flow.Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        nvp=args.nvp,
        num_classes=num_classes,
    ).cuda()
    for p in model.parameters():
        p.requires_grad = False

    model_name = f"{args.patch_size}_{args.channels}_{args.blocks}_{args.layers_per_block}_{args.noise_std:.2f}"
    sample_dir: pathlib.Path = args.logdir / f"{args.dataset}_samples_{model_name}"

    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt_file, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    print("Starting sampling")
    num_batches = int(np.ceil(args.num_samples / args.batch_size))
    last_batch_size = args.num_samples - (num_batches - 1) * args.batch_size

    def get_noise(b):
        return torch.randn(
            b,
            (args.img_size // args.patch_size) ** 2,
            args.channel_size * args.patch_size**2,
            device="cuda",
        )

    for i in range(num_batches):
        noise = get_noise(args.batch_size // dist.world_size)
        if num_classes:
            y = torch.randint(
                num_classes, (args.batch_size // dist.world_size,), device="cuda"
            )
        else:
            y = None
        while True:
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                samples = model.reverse(
                    noise, y, args.cfg, attn_temp=args.attn_temp, annealed_guidance=True
                )
                assert isinstance(samples, torch.Tensor)

            if args.self_denoising_lr > 0:
                samples = samples.cpu()
                assert args.batch_size % args.denoising_batch_size == 0
                db = args.denoising_batch_size // dist.world_size
                # This should be the theoretical optimal denoising lr
                base_lr = db * args.img_size**2 * args.channel_size * args.noise_std**2
                lr = args.self_denoising_lr * base_lr
                denoised_samples = []
                for j in range(args.batch_size // args.denoising_batch_size):
                    x = torch.clone(samples[j * db : (j + 1) * db]).detach().cuda()
                    x.requires_grad = True
                    y_ = y[j * db : (j + 1) * db] if y is not None else None
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z, _, logdets = model(x, y_)
                    loss = model.get_loss(z, logdets)
                    grad = torch.autograd.grad(loss, [x])[0]
                    x.data.add_(grad, alpha=-lr)
                    denoised_samples.append(x.detach().cpu())
                samples = torch.cat(denoised_samples, dim=0).cuda()

            samples = dist.gather_concat(samples.detach())
            if not samples.isnan().any().item():
                break
            else:
                noise = get_noise(args.batch_size // dist.world_size)

        if i == num_batches - 1:
            samples = samples[:last_batch_size]

        fid.update(0.5 * (samples.clip(min=-1, max=1) + 1), real=False)
        print(f"{i+1}/{num_batches} batch sample complete")
    fid_score = fid.compute().item()
    fid.reset()

    print(f"{args.ckpt_file} {model_name} cfg {args.cfg:.2f} fid {fid_score:.2f}")
    if dist.local_rank == 0:
        tv.utils.save_image(
            samples,
            sample_dir / f"samples_cfg{args.cfg:.2f}.png",
            normalize=True,
            nrow=16,
        )
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data", type=pathlib.Path, help="Path for training data"
    )
    parser.add_argument(
        "--logdir", default="runs", type=pathlib.Path, help="Path for artifacts"
    )

    parser.add_argument(
        "--ckpt_file", default="", type=str, help="Path for checkpoint for evaluation"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        type=str,
        choices=["imagenet", "imagenet64", "afhq"],
        help="Name of dataset",
    )
    parser.add_argument("--img_size", default=32, type=int, help="Image size")
    parser.add_argument(
        "--channel_size", default=3, type=int, help="Image channel size"
    )

    parser.add_argument(
        "--patch_size", default=4, type=int, help="Patch size for the model"
    )
    parser.add_argument("--channels", default=512, type=int, help="Model width")
    parser.add_argument(
        "--blocks", default=4, type=int, help="Number of autoregressive flow blocks"
    )
    parser.add_argument(
        "--layers_per_block", default=8, type=int, help="Depth per flow block"
    )
    parser.add_argument(
        "--noise_std", default=0.05, type=float, help="Input noise standard deviation"
    )
    parser.add_argument(
        "--nvp",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use the non volume preserving version",
    )
    parser.add_argument(
        "--cfg",
        default=0,
        type=float,
        help="Guidance weight for sampling, 0 is no guidance. For conditional models consider the range in [1, 3]",
    )
    parser.add_argument(
        "--attn_temp",
        default=1.0,
        type=float,
        help="Attention temperature for unconditional guidance, enabled when not 1 (eg, 0.5, 1.5)",
    )
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for drawing samples"
    )
    parser.add_argument(
        "--num_samples", default=50000, type=int, help="Number of total samples to draw"
    )
    parser.add_argument(
        "--self_denoising_lr",
        default=1.0,
        type=float,
        help="Learning rate multiplier for denoising, 1 is the theoretical optimal one",
    )

    args = parser.parse_args()

    main(args)
