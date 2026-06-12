from abc import abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DiffusionSchedule(nn.Module):
    @abstractmethod
    def alpha_t(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the fraction of the original signal remaining at time t
        t: [bsz]
        output shape should be [bsz, 1] or [bsz, seq_len]
        """
        raise NotImplementedError("alpha_t method must be implemented")

    @abstractmethod
    def deriv(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the derivative of alpha_t(t) w.r.t. t
        """
        raise NotImplementedError("deriv method must be implemented")

    def cross_entropy_loss_weight(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the weight for the cross entropy loss at time t
        calculated as -1 * d/dt alpha_t(t) / (1 - alpha_t(t))
        """
        return -self.deriv(t, **kwargs) / (1 - self.alpha_t(t, **kwargs))

    def p_t(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        vocab_size: int = None,
        **alpha_t_kwargs,
    ) -> Categorical:
        """
        x.shape = [bsz, seq_len, vocab_size]
        t.shape = [bsz] (or float scalar)
        """
        if x_0.ndim == 2:
            assert (
                vocab_size is not None
            ), "vocab_size must be provided for 2D (categorical) input"
            bsz, seq_len = x_0.shape
            x_0 = F.one_hot(x_0, num_classes=vocab_size)
        elif x_0.ndim == 3:
            bsz, seq_len, vocab_size = x_0.shape
        else:
            raise ValueError(f"{x_0.ndim=}. x_0 should have 2 or 3 dimensions")
        if type(t) == float:
            t = torch.full((bsz,), t, device=x_0.device)
        mu_t = torch.zeros((bsz, seq_len, vocab_size + 1), device=x_0.device)
        alpha_t = self.alpha_t(t, **alpha_t_kwargs).detach()
        # TODO: the following could be simplified by using a one-hot to represent the mask
        # signal component
        mu_t[..., :-1] = x_0 * alpha_t[..., None]
        # noise component
        mu_t[..., -1] += 1 - alpha_t
        return Categorical(probs=mu_t)


class CosineDiffusionSchedule(DiffusionSchedule):
    @staticmethod
    def alpha_t(t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos(math.pi / 2 * (1 - t)).unsqueeze(1)

    @staticmethod
    def cross_entropy_loss_weight(t: torch.Tensor) -> torch.Tensor:
        # Table 4 of https://arxiv.org/abs/2406.04329
        return (
            (math.pi / 2 * torch.tan(math.pi / 2 * (1 - t)))
            .clamp(min=1e-2, max=1e2)
            .unsqueeze(1)
        )

    @staticmethod
    def deriv(t: torch.Tensor) -> torch.Tensor:
        return -math.pi / 2 * torch.sin(math.pi / 2 * (1 - t)).unsqueeze(1)
