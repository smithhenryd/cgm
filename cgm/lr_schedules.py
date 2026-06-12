import math

import torch
from torch import optim


def warmup_cosine_multiplier(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """
    Piecewise LR multiplier with linear warmup then cosine decay.

    The returned value multiplies the optimizer base learning rate:
      - warmup: linearly ramps from 1/warmup_epochs to 1.0
      - decay:  cosine decays from 1.0 to min_lr_ratio
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if not (0.0 < min_lr_ratio <= 1.0):
        raise ValueError("min_lr_ratio must be in (0, 1]")

    warmup_epochs = min(warmup_epochs, total_epochs - 1)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs

    if total_epochs - warmup_epochs <= 1:
        return 1.0

    decay_step = epoch - warmup_epochs
    decay_steps = total_epochs - warmup_epochs - 1
    progress = min(max(decay_step / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def make_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> optim.lr_scheduler.LambdaLR:
    """
    Construct a LambdaLR scheduler using warmup_cosine_multiplier.
    """

    return optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_multiplier(
            epoch=epoch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )
