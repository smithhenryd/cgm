import time
import math
import warnings
from dataclasses import is_dataclass
from typing import Any

import torch
from torch import optim
from torch.optim import Optimizer
from pytorch_lightning import LightningModule

from torch.optim.lr_scheduler import _LRScheduler


def pick_precision() -> str:
    if torch.cuda.get_device_capability()[0] == 8:
        return "bf16-mixed"
    return "32"


def dataclass_get(obj: Any, key: str, default: Any) -> Any:
    """
    Safely retrieve an attribute from a dataclass with a fallback default.

    If the attribute is missing or set to None, this function:
    - Returns the provided default value
    - Sets the attribute on the dataclass to the default value

    Parameters:
        obj (Any): The dataclass instance.
        key (str): The name of the attribute to retrieve.
        default (Any): The value to return and set if the attribute is missing or None.

    Returns:
        Any: The attribute value if present and not None; otherwise, the default.

    Raises:
        TypeError: If the input object is not a dataclass instance.
    """
    if not is_dataclass(obj):
        raise TypeError(f"Expected a dataclass instance, got {type(obj)}")

    val = getattr(obj, key, None)
    if val is None:
        setattr(obj, key, default)
        return default
    return val


class BaseLightning(LightningModule):
    """
    Expects self.hparams.cfg to be a dataclass!
    optional hparams:
    * lr
    * wd
    * print_freq
    """

    @staticmethod
    def get_loss(batch, prefix: str) -> torch.Tensor:
        pass

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.get_loss(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], *args
    ) -> torch.Tensor:
        return self.get_loss(batch, "val")

    def smart_log(self, name: str, value, prefix: str):
        name = f"{prefix}/{name}"
        if prefix == "train":
            self.log(name, value, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_time)
        print_freq = dataclass_get(self.hparams.cfg, "print_freq", 1)
        if self.current_epoch % print_freq == 0:
            print(f"train epoch {self.current_epoch} took {epoch_time:.1f}s")

    def on_validation_epoch_start(self):
        self.val_epoch_start_time = time.time()

    def on_validation_epoch_end(self):
        epoch_time = time.time() - self.val_epoch_start_time
        self.log("epoch_time", epoch_time)
        print_freq = dataclass_get(self.hparams.cfg, "print_freq", 1)
        if self.current_epoch % print_freq == 0:
            print(f"val epoch {self.current_epoch} took {epoch_time:.1f}s")

    def configure_optimizers(self):
        # Define optimizer
        lr = dataclass_get(self.hparams.cfg, "lr", 5e-4)
        wd = dataclass_get(self.hparams.cfg, "wd", 0)

        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        # Use the Trainer's max_epochs to define the scheduler's max_epochs
        if self.trainer is None:
            raise RuntimeError(
                "Trainer must be attached before calling configure_optimizers"
            )

        max_epochs = self.trainer.max_epochs
        warmup_epochs = min(max(1, max_epochs // 20), 10)
        total_batches = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_batches // max_epochs
        print(f"Estimated steps per epoch: {steps_per_epoch}")

        # Define scheduler
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=warmup_epochs * steps_per_epoch,
            max_epochs=max_epochs * steps_per_epoch,
            warmup_start_lr=lr / 100,
            eta_min=lr / 100,
        )

        # Return optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'step', based on your preference
                "frequency": 1,
            },
        }


# from pl_bolts https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/optimizers/lr_scheduler.py
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.

    Example:
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> #
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)

    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]
