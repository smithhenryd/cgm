from typing import Callable, Iterable

import torch


Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def energy_distance_kernel(
    p: float = 2.0, beta: float = 1.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Return the kernel k(x, y) = -||x - y||_p^beta.

    With this kernel MMD^2(P, Q) equals the energy distance ED(P, Q).
    """

    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -(torch.cdist(x, y, p=p) ** beta)

    return kernel


def rbf_mixture_kernel(
    sigmas: Iterable[float] | torch.Tensor = (0.08, 0.16, 0.32, 0.64),
) -> Kernel:
    """
    Return an RBF mixture kernel k(x, y) = mean_s exp(-||x-y||^2 / (2 sigma_s^2)).

    Args:
        sigmas: Bandwidth values (on the same scale as the inputs).
    """
    _sigmas = sigmas.tolist() if torch.is_tensor(sigmas) else list(sigmas)

    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x, y, p=2.0) ** 2  # [n, m]
        s = torch.tensor(_sigmas, device=x.device, dtype=x.dtype)  # [S]
        K = torch.exp(-d2.unsqueeze(-1) / (2.0 * s.view(1, 1, -1) ** 2))  # [n, m, S]
        return K.mean(dim=-1)  # [n, m]

    return kernel


def dot_product_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Linear kernel k(x, y) = x @ y.T.

    With this kernel, MMD^2(P, Q) is the squared Euclidean distance between
    the feature means E_P[X] and E_Q[Y].
    """
    return x @ y.T


def median_heuristic_rbf_sigmas(
    hstar: torch.Tensor,
    factors: Iterable[float] | torch.Tensor = (0.25, 0.5, 1.0, 2.0, 4.0),
    max_samples: int | None = 1000,
) -> torch.Tensor:
    """
    Choose an RBF bandwidth grid from target feature samples.

    The grid is centered at the median pairwise Euclidean distance of hstar,
    optionally after random subsampling, and scaled by the provided factors.

    Args:
        hstar: Target feature samples with shape [m, d].
        factors: Multipliers applied to the median pairwise distance.
        max_samples: If set and m > max_samples, estimate the median from a
            random subset of this many samples.
    """
    if hstar.ndim != 2:
        raise ValueError(f"hstar must have shape [m, d], got {tuple(hstar.shape)}")
    if hstar.shape[0] < 2:
        raise ValueError("Need at least two target samples to estimate RBF sigmas.")

    hstar_sub = hstar
    if max_samples is not None and hstar.shape[0] > max_samples:
        idx = torch.randperm(hstar.shape[0], device=hstar.device)[:max_samples]
        hstar_sub = hstar[idx]

    median_dist = torch.median(torch.pdist(hstar_sub, p=2.0))
    scales = torch.as_tensor(factors, device=hstar.device, dtype=hstar.dtype)
    eps = torch.finfo(hstar.dtype).eps
    return median_dist.clamp_min(eps) * scales


def tanimoto_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Tanimoto (Jaccard) kernel for binary fingerprints.

    This is a positive semi-definite kernel, making MMD^2 with it a valid
    divergence. It is the standard similarity measure for Morgan fingerprints
    and requires no bandwidth hyperparameters.

    Args:
        x: [n, d] binary fingerprint vectors (float).
        y: [m, d] binary fingerprint vectors (float).

    Returns:
        [n, m] kernel matrix with values in [0, 1].
    """
    intersection = x @ y.T  # [n, m]
    union = x.sum(-1, keepdim=True) + y.sum(-1, keepdim=True).T - intersection
    return intersection / union.clamp_min(1e-8)


def scale_kernel(kernel: Kernel, scale: float) -> Kernel:
    """
    Useful for numerics, since some kernels have very small values.
    """
    return lambda x, y: scale * kernel(x, y)
