#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
__all__ = [
    "CosineLRSchedule",
    "Distributed",
    "FID",
    "Metrics",
    "get_data",
    "set_random_seed",
]

import datetime
import math
import os
import pathlib
import random

import numpy as np
import torch
import torch.distributed
import torch.utils.data
import torchvision as tv
from torchmetrics.image.fid import FrechetInceptionDistance


class CosineLRSchedule(torch.nn.Module):
    counter: torch.Tensor

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float,
        max_lr: float,
    ):
        super().__init__()
        self.register_buffer("counter", torch.zeros(()))
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.set_lr(min_lr)

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        return max(self.min_lr, min(self.max_lr, lr))

    def step(self) -> float:
        with torch.no_grad():
            counter = self.counter.add_(1).item()
        if self.counter <= self.warmup_steps:
            new_lr = self.min_lr + counter / self.warmup_steps * (
                self.max_lr - self.min_lr
            )
            return self.set_lr(new_lr)

        t = (counter - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        new_lr = self.min_lr + 0.5 * (1 + math.cos(math.pi * t)) * (
            self.max_lr - self.min_lr
        )
        return self.set_lr(new_lr)


class Distributed:

    def __init__(self):
        if os.environ.get("MASTER_PORT"):  # When running with torchrun
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.distributed = True
            torch.distributed.init_process_group(
                "nccl", "env://", timeout=datetime.timedelta(minutes=10)
            )
        else:  # When running with python for debugging
            self.rank, self.local_rank, self.world_size = 0, 0, 1
            self.distributed = False
        torch.cuda.set_device(self.local_rank)
        self.barrier()

    def barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier()

    def gather_concat(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed:
            return x
        x_list = [torch.empty_like(x) for _ in range(self.world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.cat(x_list)

    def __del__(self):
        if self.distributed:
            torch.distributed.destroy_process_group()


class FID(FrechetInceptionDistance):
    def add_state(self, name, default, *args, **kwargs):
        self.register_buffer(name, default)


class Metrics:
    def __init__(self):
        self.metrics: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, torch.Tensor | float]):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def compute(self, dist: Distributed | None) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.metrics.items():
            v = sum(v) / len(v)
            if dist is not None:
                v = (
                    dist.gather_concat(torch.tensor(v, device="cuda").view(1))
                    .mean()
                    .item()
                )
            out[k] = v
        return out

    @staticmethod
    def print(metrics: dict[str, float], epoch: int):
        print(f"Epoch {epoch}  Time {datetime.datetime.now()}")
        print("\n".join((f"\t{k:40s}: {v: .4g}" for k, v in sorted(metrics.items()))))


def get_num_classes(dataset: str) -> int:
    return {"imagenet64": 0, "imagenet": 1000, "afhq": 3}[dataset]


def get_data(
    dataset: str, img_size: int, folder: pathlib.Path
) -> tuple[torch.utils.data.Dataset, int]:
    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(img_size),
            tv.transforms.CenterCrop(img_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if dataset == "imagenet64":
        data = tv.datasets.ImageFolder(str(folder / "imagenet64"), transform=transform)
    elif dataset == "imagenet":
        data = tv.datasets.ImageFolder(str(folder / "imagenet"), transform=transform)
    elif dataset == "afhq":
        data = tv.datasets.ImageFolder(str(folder / "afhq"), transform=transform)
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")
    return data, get_num_classes(dataset)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
