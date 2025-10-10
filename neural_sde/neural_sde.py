from typing import Any, Callable, Optional

from cgm.model import Model
from cgm.utils import chunk_bounds, module_device

import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class SamplePath:
    """
    Sample paths drawn from a Neural-SDE

    xs: a shape (batch_size, Nsteps, sde_dim) tensor, batch_size sample paths drawn from the solution to the Neural-SDE
    zs: a shape (batch_size, Nsteps - 1, sde_dim) tensor, the Brownian motion paths used for sampling the solution
    ts: a shape (Nsteps) tensor, the discrete time grid on which the sample paths are evaluated
    """

    xs: torch.Tensor
    zs: torch.Tensor
    ts: torch.Tensor


class NeuralSDE(Model[SamplePath]):

    def __init__(
        self,
        sde_dim: int,
        drift: nn.Module,
        diffusion: Callable[[torch.Tensor], torch.Tensor],
        t_grid: torch.Tensor,
        initial_dist: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:

        super().__init__()
        self.sde_dim = sde_dim
        self.drift = drift
        self.diffusion = diffusion
        self.t_grid = t_grid
        self.sample_x0 = (
            initial_dist
            if (initial_dist is not None)
            else (
                lambda N: torch.normal(
                    torch.zeros((N, self.sde_dim), device=self.drift.device), 1.0
                )
            )
        )

    def sample(self, N: int) -> SamplePath:

        # First draw samples from the initial distribution
        x0 = self.sample_x0(N)

        xt = x0
        Nsteps = self.t_grid.shape[0] - 1
        xts = torch.zeros((N, Nsteps + 1, self.sde_dim), device=self.drift.device)
        xts[:, 0] = xt
        zs = torch.zeros((N, Nsteps, self.sde_dim), device=self.drift.device)
        for i in range(Nsteps):
            del_t = self.t_grid[i + 1] - self.t_grid[i]

            # Compute drift and diffusion
            drift_term = self.drift(xt, self.t_grid[i].repeat(N))
            diffusion_term = self.diffusion(self.t_grid[i].repeat(N))

            # Euler-Maruyama update, storing the noise
            noise_std = torch.sqrt(del_t) * diffusion_term
            noise = noise_std[:, None] * torch.normal(
                0, 1.0, size=xt.shape, device=self.drift.device
            )

            xt = xt + del_t * drift_term + noise

            xts[:, i + 1] = xt
            zs[:, i] = (
                noise / diffusion_term[:, None]
            )  # NOTE: noise is still scaled by \sqrt{del_t}

        return SamplePath(xts, zs, self.t_grid)

    def log_p(
        self,
        x: SamplePath,
        batch_idx: int = 0,
        batch_chunks: int = 1,
        sample_idx: int = 0,
        sample_chunks: int = 1,
    ) -> torch.Tensor:
        """
        Computes the log RN derivative log(dP/dQ), for samples x drawn from Q
        """

        x = self._extract_chunk(x, batch_idx, batch_chunks, sample_idx, sample_chunks)
        xs, zs, ts = x.xs, x.zs, x.ts
        batch_size, Nsteps, sde_dim = xs.shape  # (batch_size, Nsteps, sde_dim)
        del_t = ts[1:] - ts[:-1]  # (Nsteps -1)

        # Compute noise
        sigmas = self.diffusion(ts[:-1]).reshape([Nsteps - 1])  # (Nsteps-1)

        # Compute the drift under Q
        fs_Q = ((xs[:, 1:, :] - xs[:, :-1, :]) - sigmas[None, :, None] * zs) / del_t[
            None, :, None
        ]  # (batch_size, Nsteps - 1, sde_dim)

        # Compute drift under P
        fs_P = self.drift(
            xs[:, :-1, :].reshape([batch_size * (Nsteps - 1), sde_dim]),
            ts[:-1].repeat(batch_size, 1).reshape([batch_size * (Nsteps - 1)]),
        ).reshape(
            [batch_size, Nsteps - 1, sde_dim]
        )  # (batch_size, Nsteps - 1, sde_dim)

        # Compute log Girsanov derivative
        return ((fs_P - fs_Q) / sigmas[None, :, None] * zs).sum(dim=[-1, -2]) - 0.5 * (
            torch.square((fs_P - fs_Q) / sigmas[None, :, None]) * del_t[None, :, None]
        ).sum(dim=[-1, -2])

    @property
    def device(self) -> torch.device:
        return module_device(self.drift)

    def _extract_chunk(
        self,
        x: SamplePath,
        batch_idx: int,
        batch_chunks: int,
        sample_idx: int,
        sample_chunks: int,
    ) -> SamplePath:
        """
        Helper function, extracts a subpath of the sample path x
        """

        batch_size, T, _ = x.xs.shape
        T -= 1

        # Batch slice
        min_i, max_i = chunk_bounds(batch_size, batch_chunks, batch_idx)

        # Time slice for xs
        min_j, max_j = chunk_bounds(T, sample_chunks, sample_idx)

        return SamplePath(
            x.xs[min_i:max_i, min_j : (max_j + 1), :],
            x.zs[min_i:max_i, min_j:max_j, :],
            x.ts[min_j : (max_j + 1)],
        )
