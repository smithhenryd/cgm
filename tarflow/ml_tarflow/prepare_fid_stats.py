#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import pathlib

import torch
import torch.utils.data

import utils


def main(args):
    dist = utils.Distributed()
    data, _ = utils.get_data(args.dataset, args.img_size, args.data)

    fid_stats_file = args.data / f"{args.dataset}_{args.img_size}_fid_stats.pth"
    assert not fid_stats_file.exists()
    fid = utils.FID(reset_real_features=False, normalize=True).cuda()
    dist.barrier()

    data_sampler = torch.utils.data.DistributedSampler(
        data, num_replicas=dist.world_size, rank=dist.local_rank, shuffle=False
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        sampler=data_sampler,
        batch_size=args.batch_size // dist.world_size,
        num_workers=8,
        drop_last=False,
    )

    for x, _ in data_loader:
        x = x.cuda()
        fid.update(dist.gather_concat(0.5 * (x + 1)), real=True)

    if dist.local_rank == 0:
        torch.save(fid.state_dict(), fid_stats_file)
        print(f"Saved FID stats file {fid_stats_file}")
    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data", type=pathlib.Path, help="Path for training data"
    )
    parser.add_argument(
        "--dataset",
        default="imagenet",
        choices=["imagenet", "imagenet64", "afhq"],
        help="Name of dataset",
    )
    parser.add_argument("--img_size", default=32, type=int, help="Image size")
    parser.add_argument(
        "--channel_size", default=3, type=int, help="Image channel size"
    )
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size")
    args = parser.parse_args()

    main(args)
