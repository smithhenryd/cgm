from typing import Any, Callable, Optional

from cgm.model import Model, module_device
from cgm.utils import chunk_bounds

import torch
import torch.nn as nn

from dataclasses import dataclass

@dataclass
class Sample:
    """
    Represents a sample drawn from a normalizing flow

    xs: a shape (batch_size, dim) tensor
    eps: a shape (batch_size, dim) tensor, noise added to the samples
    NOTE: eps should be set to zero if the normalizing flow is trained directly on the data distribution
    """
    xs: torch.Tensor
    eps: torch.Tensor


class NormalizingFlow(Model[Sample]):

    def __init__(
        self,
        dim: int,
        map: nn.Module,
        initial_dist: Optional[Callable[[int], torch.Tensor]] = None,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.map = map
        self.initial_dist = (
            initial_dist
            if (initial_dist is not None)
            else (
                lambda N: torch.normal(
                    torch.zeros((N, self.dim), device=self.device), 1.0
                )
            )
        )

    def sample(self, N: int, **kwargs) -> Sample:

        # Draw N samples from the initial distribution
        zs = self.initial_dist(N)

        # Map the samples through the map
        xs, eps = self.map.reverse(zs, **kwargs)
        return Sample(xs, eps)

    def log_p(
        self, x: Sample, batch_idx: int = 0, batch_chunks: int = 1, **kwargs
    ) -> torch.Tensor:

        x = self._extract_chunk(x, batch_idx, batch_chunks)
        return self.map.log_p(x.xs, x.eps)  # (batch_size)

    @property
    def device(self) -> torch.device:
        return module_device(self.map)

    def _extract_chunk(
        self,
        x: Sample,
        batch_idx: int,
        batch_chunks: int,
    ) -> Sample:
        """
        Helper function, extracts a subset of the samples x (i.e. along the batch dimension)
        """

        # Batch slice
        batch_size, _ = x.xs.shape
        min_i, max_i = chunk_bounds(batch_size, batch_chunks, batch_idx)

        return Sample(x.xs[min_i:max_i], x.eps[min_i:max_i])
