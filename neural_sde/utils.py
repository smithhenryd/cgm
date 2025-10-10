import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
import os

from tqdm import tqdm

from typing import Any, Callable, Optional, Type, Tuple, Union

from cgm.utils import default_logger, module_device
from neural_sde import NeuralSDE, SamplePath


# -------------------------- Neural-SDE training --------------------------
def kappa(t: float, eps: float = 1e-2):
    """
    Parameterizes the OU forward noising process
    dx(t) = -(1/2) \kappa(t) x(t) dt + \sqrt{\kappa(t)} dw(t), x(0) fixed, 0 <= t <= 1

    Produces the marginals p(x(t) | x(0)) = N(x(t) | (1-t)x(0), t*I) (when eps = 0)
    """
    return 1 / (1 - t + eps)


def m(t):
    """
    Conditional mean of the forward process E[x(t) | x(0)]
    """
    return torch.sqrt(1 - t)


def sigma(t):
    """
    Conditional std of the forward process (Var[x(t) | x(0)])^{1/2}
    """
    return torch.sqrt(t)


def train_neural_sde(
    x0s: torch.Tensor,
    score_net: nn.Module,
    mean_fn: Callable[[float], float] = m,
    std_fn: Callable[[float], float] = sigma,
    llambda: Callable[[float], float] = sigma,
    epochs: int = 100,
    batches: int = 500,
    batch_size: int = 256,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-2},
    lr_scheduler_cls: Optional[
        Type[optim.lr_scheduler.LRScheduler]
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    logger: Callable[[dict[str, Any]], None] = default_logger,
    jitter: float = 1e-5,
) -> nn.Module:
    """
    Simple score-matching trainer for neural SDEs (Song et al., "Score-based gradient generative modeling
    through stochastic differential equations". ICLR, 2021.)

    Assumes p(x_t | x_0) = N(m(t) * x_0, [s(t)]^2 I), and the score target is
    ∇_{x_t} log p(x_t | x_0) = -(x_t - m(t) x_0)/s(t)^2 = -eps / s(t),
    where x_t = m(t) x_0 + s(t) * eps
    """
    # SETUP
    Nsamps, D = x0s.shape
    device = score_net.device

    ## Optimizer
    optimizer = optimizer_cls(score_net.parameters(), **optimizer_params)

    ## Learning rate scheduler
    scheduler_params = (
        {"T_max": epochs, "eta_min": 1e-5}
        if (
            scheduler_params is None
            and lr_scheduler_cls is optim.lr_scheduler.CosineAnnealingLR
        )
        else scheduler_params
    )
    scheduler = (
        lr_scheduler_cls(optimizer, **scheduler_params)
        if lr_scheduler_cls is not None
        else None
    )

    # TRAINING LOOP
    pbar = tqdm(range(epochs), desc="Training Epochs")
    for epoch in pbar:

        total_loss = 0.0
        for _ in range(batches):

            ## Sample t uniformly from [0, 1]
            t = torch.rand(batch_size, device=device)

            ## Select samples to noise
            idx = torch.randint(Nsamps, (batch_size,), device=device)

            ## Noise the samples
            m, s = mean_fn(t), std_fn(t)  # (batch_size)
            eps = torch.normal(
                torch.zeros(batch_size, D, device=device), 1.0
            )  # (batch_size, D)
            xt = x0s[idx] * m[:, None] + s[:, None] * eps

            ## Predict the score using the score network
            predicted_score = score_net(xt, t.unsqueeze(-1))  # (batch_size, D)
            ## Compute the true conditional score
            true_score = -eps / (s[:, None] + jitter)  # (batch_size, D)

            ## Compute the weighting function λ(t) for score matching loss
            weights = llambda(t)  # (batch_size)
            ## MSE loss
            loss = (
                weights[:, None]
                * F.mse_loss(predicted_score, true_score, reduction="none")
            ).mean()

            ## Update model parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch == 0 and total_loss == 0:
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{total_loss:.4f}"})

            total_loss += loss.item()

        ## Log the total score matching loss
        total_loss = total_loss / batches
        pbar.set_postfix({"loss": f"{total_loss:.4f}"})

        if logger is not None:
            logger({"epoch": epoch, "loss": total_loss})

        scheduler.step() if scheduler is not None else None

    # RETURN THE LEARNED SCORE NETWORK
    return score_net


# -------------------------- Drift functions --------------------------
def sinusoidal_time_embedding(t: float, embedding_dim: int):
    """
    Computes a sinusoidal (Fourier) embedding of the scalar time t
    """
    half_dim = embedding_dim // 2
    device = t.device
    freq = (
        2
        * torch.pi
        * (4 * (torch.arange(half_dim, dtype=t.dtype, device=device) + 1))
        / half_dim
    )
    scaled_t = t * freq.unsqueeze(0)

    # Sinusoidal features: cat(sin, cos)
    sin_emb = torch.sin(scaled_t)
    cos_emb = torch.cos(scaled_t)

    return torch.cat([sin_emb, cos_emb], dim=-1)


class BackwardDrift(nn.Module):
    """
    Class representing the learned drift of a neural-SDE from the forward drift and diffusion functions as well as the learned score of the forward process

    We assume the diffusion coefficient has the form σ(t) I, where σ depends only on time
    """

    def __init__(
        self,
        score: nn.Module,
        drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        score: a neural network representing the learned score of the forward noising process
        drift: the drift of the forward process
        diffusion: the diffusion coefficient of the forward process
        """
        super().__init__()
        self.score = score
        self.drift = drift
        self.diffusion = diffusion

    @property
    def device(self) -> torch.device:
        return module_device(self.score)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f = self.drift(x, 1.0 - t)
        s = self.diffusion(1.0 - t)
        return s[:, None] ** 2 * self.score(x, (1.0 - t).unsqueeze(-1)) - f


class ScoreNetwork(nn.Module):
    """
    Simple NN architecture for modelling the score of a noising  process
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        sigma: Callable[[torch.Tensor], torch.Tensor] = sigma,
        device: Type[torch.device] = torch.device("cpu"),
    ) -> None:

        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.sigma = sigma

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.to(self.device)

    def score(
        self, x: torch.Tensor, t: torch.Tensor, jitter: float = 1e-5
    ) -> torch.Tensor:
        # Embed time t with sinusoidal features
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)

        # Concatenate x and t-embedding
        inp = torch.cat([x, t_emb], dim=-1)

        # Noise prediction
        eps_hat = self.net(inp)

        # Convert to a score prediction
        s = self.sigma(t.squeeze(-1))
        out = -eps_hat / (s[:, None] + jitter)
        return out

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(x, t)


# -------------------------- KL estimate --------------------------
def compute_KL_est(base_model: NeuralSDE, x: SamplePath) -> torch.Tensor:
    """
    Computes the reduced variance KL estimate (i.e. drop the SI term in the log density ratio)
    """

    xs, zs, ts = x.xs, x.zs, x.ts
    batch_size, Nsteps, sde_dim = xs.shape  # (batch_size, Nsteps, sde_dim)
    del_t = ts[1:] - ts[:-1]  # (Nsteps -1)

    # Compute noise
    sigmas = base_model.diffusion(ts[:-1]).reshape([Nsteps - 1])  # (Nsteps-1)

    # Compute the drift under p_{\theta}
    fs_Q = ((xs[:, 1:, :] - xs[:, :-1, :]) - sigmas[None, :, None] * zs) / del_t[
        None, :, None
    ]  # (batch_size, Nsteps - 1, sde_dim)

    # Compute the drift under p_{base}
    fs_P = base_model.drift(
        xs[:, :-1, :].reshape([batch_size * (Nsteps - 1), sde_dim]),
        ts[:-1].repeat(batch_size, 1).reshape([batch_size * (Nsteps - 1)]),
    ).reshape([batch_size, Nsteps - 1, sde_dim])
    kl_est = 0.5 * (
        torch.square((fs_P - fs_Q) / sigmas[None, :, None]) * del_t[None, :, None]
    ).sum(dim=[-1, -2])
    return kl_est


# -------------------------- Save and load a Neural-SDE object --------------------------
def save_checkpoint(
    model: NeuralSDE,
    ckpt_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
    epoch: int = 0,
) -> str:
    """
    Save a NeuralSDE checkpoint
    """
    ckpt_dir = os.fspath(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")

    blob = {
        "epoch": int(epoch),
        "sde_dim": int(getattr(model, "sde_dim", 0)),  # good to save for sanity checks
        "t_grid": model.t_grid.detach().clone(),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        blob["optim_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        blob["scheduler_state"] = scheduler.state_dict()

    torch.save(blob, ckpt_path)
    return ckpt_path


def load_checkpoint(
    ckpt_path: Union[str, Path],
    model: NeuralSDE,
    map_location: Union[torch.device, str] = "auto",
    strict: bool = True,
) -> Tuple[NeuralSDE, Optional[dict], Optional[dict], int]:
    """
    Load a NeuralSDE checkpoint
    """

    ckpt_path = os.fspath(ckpt_path)

    # Resolve target device
    if map_location == "auto":
        try:
            target = model.device
        except Exception:
            target = torch.device("cpu")
    else:
        target = (
            torch.device(map_location)
            if isinstance(map_location, str)
            else map_location
        )

    # Ensure the model is on the target device before loading weights
    model.to(target)

    # Load blob with tensors mapped to target
    blob = torch.load(ckpt_path, map_location=target)

    # Check that the dimension of the SDE lines up with the saved model
    if "sde_dim" in blob:
        ckpt_dim = int(blob["sde_dim"])
        if strict and hasattr(model, "sde_dim") and int(model.sde_dim) != ckpt_dim:
            raise RuntimeError(
                f"SDE dimension mismatch: model={int(model.sde_dim)} vs ckpt={ckpt_dim}"
            )

    # Put t_grid onto target device
    t_grid = blob["t_grid"]
    if not isinstance(t_grid, torch.Tensor):
        raise RuntimeError("Checkpoint field 't_grid' is not a torch.Tensor")
    model.t_grid = t_grid

    # Load weights
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Error loading NeuralSDE checkpoint with strict=True:\n"
            f"  Missing keys:    {missing}\n"
            f"  Unexpected keys: {unexpected}"
        )

    optim_state = blob.get("optim_state")
    scheduler_state = blob.get("scheduler_state")
    epoch = int(blob["epoch"])
    print(f"Successfully checkpoint {ckpt_path}")
    return model, optim_state, scheduler_state, epoch
