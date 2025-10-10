import os
import copy
import warnings
from pathlib import Path
from abc import abstractmethod, ABC
import sys
from collections import defaultdict
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import optim
import cvxpy as cp

from typing import Any, TextIO, Union, Optional


def clone_network(net: nn.Module, disable_gradients=True) -> nn.Module:
    """
    Clone a torch.nn.Module such that the cloned model does not track gradients
    """

    # Deep copy preserves architecture and parameters
    cloned_net = copy.deepcopy(net)

    # Disable gradients for all parameters
    if disable_gradients:
        for param in cloned_net.parameters():
            param.requires_grad = False

    return cloned_net


def module_device(m: nn.Module) -> torch.device:
    """
    A helper function; extracts the device of a torch.nn.Module

    Assumes all model parameters are on the same device
    """
    for p in m.parameters():
        return p.device
    for b in m.buffers():
        return b.device
    return torch.device("cpu")


def default_logger(
    logs: dict[str, Any],
    dest: Union[TextIO, str, Path] = sys.stdout,
) -> None:
    """
    Prints a logging dict as 'key=value' pairs, comma-separated, with floats at 4 decimals to the specified destination
    """

    def to_scalar(v: Any) -> Any:
        # Convert 0D tensors to Python scalars, if possible
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return v.item()
        return v

    def fmt(v: Any) -> str:
        v = to_scalar(v)
        if isinstance(v, int):
            return str(v)
        elif isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    line = ", ".join(f"{k}={fmt(v)}" for k, v in logs.items())

    if isinstance(dest, (str, os.PathLike, Path)):  # write output to a file
        with open(dest, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:  # write output to IO
        dest.write(line + "\n")
        if hasattr(dest, "flush"):
            dest.flush()


class DictLogger:
    def __init__(self):
        self.metrics = defaultdict(list)

    def __call__(self, logs: dict[str, Any], *args):
        for k, v in logs.items():
            self.metrics[k].append(v)


def chunk_bounds(n: int, k: int, i: int):
    """
    Evenly partitions range(n) into k chunks
    Let q = n//k and r = n%k; then the first r chunks have size q+1, the remaining k-r chunks have size q
    """
    q, r = divmod(n, k)
    start = i * q + min(i, r)
    end = start + q + (1 if i < r else 0)
    return start, end


def compute_violation_loss(
    hx: torch.Tensor, h_target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Computes the violation loss from h(x), h*, and weights
    """
    hx = hx - h_target[None, :]

    # Multiply by weights to track gradients
    hx = hx * weights[:, None]
    hbar = torch.mean(hx, dim=0)

    # Error estimate, and bias correction
    sqr_err = hbar**2
    bias_correction = -((hx - hbar[None, :]) ** 2).sum(dim=0) / (
        hx.shape[0] * (hx.shape[0] - 1)
    )

    return sqr_err.sum() + bias_correction.sum()


def solve_dual(
    H: torch.Tensor, h_target: torch.Tensor, max_iters: int = 2500
) -> torch.Tensor:
    """
    Solves the maximum entropy problem for the empirical measure

    H: matrix of shape (M, N), where M is the number of samples and N is the
    number of moment constraints
    h_target: a shape (N) vector of moment constraints
    """
    device = H.device
    H = H.cpu().numpy()  # put everything on CPU for the dual solve
    h_target = h_target.cpu().numpy()

    N = H.shape[-1]
    alpha = cp.Variable(N)
    dual_obj = cp.Maximize(alpha @ h_target - cp.log_sum_exp(H @ alpha))

    dual_prob = cp.Problem(dual_obj)
    with warnings.catch_warnings():  # ignore unused sparse matrix warning1
        warnings.filterwarnings(action="ignore")
        dual_prob.solve(
            solver=cp.SCS, max_iters=max_iters
        )  # use the splitting conic solver (SCS)
    if alpha.value is None or "optimal" not in dual_prob.status:
        raise ValueError(f"Dual is not feasible: {dual_prob.status=}")
    return torch.tensor(alpha.value, device=device)


def default_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--lambd",
        type=float,
        required=False,
        help="Regularization strength (float).",
        default=0.1,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Number of samples per batch (default: 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        required=False,
        help="Learning rate (float).",
    )
    parser.add_argument(
        "--calibration_mode",
        type=str,
        default="relax",
        choices=["relax", "reward", "forward_kl"],
    )
    return parser


@dataclass
class CheckpointFn(ABC):
    ckpt_dir: str

    def __post_init__(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)

    @abstractmethod
    def __call__(
        self,
        model: nn.Module,
        loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> Optional[str]:
        pass

    def checkpoint_name(self, loss: float, epoch: int) -> str:
        return f"ckpt_epoch_{epoch}.pth"

    def save_checkpoint(
        self,
        model: nn.Module,
        loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> str:
        checkpoint_name = self.checkpoint_name(loss, epoch)
        ckpt_path = os.path.join(self.ckpt_dir, checkpoint_name)
        blob = {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            blob["optim_state"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            blob["scheduler_state"] = scheduler.state_dict()

        torch.save(blob, ckpt_path)
        return ckpt_path


@dataclass
class BestCheckpoint(CheckpointFn):
    verbose: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.min_loss = float("inf")

    def checkpoint_name(self, *args, **kwargs) -> str:
        return "best_checkpoint.pth"

    def __call__(
        self,
        model: nn.Module,
        loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> Optional[str]:
        if loss < self.min_loss:
            if self.verbose:
                print(
                    f"Loss improved from {self.min_loss:.3f} to {loss:.3f} on step {epoch}",
                    flush=True,
                )
            self.min_loss = loss
            return self.save_checkpoint(
                model=model,
                loss=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


@dataclass
class CheckpointEveryN(CheckpointFn):
    N: int

    def __call__(
        self,
        model: nn.Module,
        loss: float,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
    ) -> Optional[str]:
        if (epoch + 1) % self.N == 0:
            return self.save_checkpoint(
                model=model,
                loss=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Union[str, torch.device] = "cpu",
    weights_only: bool = True,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=weights_only)

    # model
    model.load_state_dict(ckpt["model_state"])

    # optimizer
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])

    # scheduler
    if scheduler is not None and "scheduler_state" in ckpt:
        # Important: load after optimizer so param groups match
        scheduler.load_state_dict(ckpt["scheduler_state"])

    # Resume training after the saved epoch
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch
