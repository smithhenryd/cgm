from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Type, TypeVar, Literal

import torch
import torch.optim as optim
from tqdm import tqdm

from cgm.model import Model
from cgm import utils
from cgm.cgm import log_p_chunked
from cgm.lr_schedules import make_warmup_cosine_scheduler

SampleType = TypeVar("SampleType")


@dataclass
class PCATransform:
    """
    A fitted PCA (optionally whitening) transform, returned by fit_pca.

    Attributes:
        mu:    [d] sample mean of the fitting data.
        evecs: [d, k] principal component vectors (columns, descending variance).
        scales:[k] per-component scale (1/sqrt(eigenvalue) if whitened, else ones).
    """

    mu: torch.Tensor
    evecs: torch.Tensor
    scales: torch.Tensor

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the transform: center, project, scale.  [..., d] -> [..., k]."""
        return ((z - self.mu) @ self.evecs) * self.scales


def fit_pca(
    y: torch.Tensor,
    n_components: int | None = None,
    whiten: bool = False,
) -> PCATransform:
    """
    Fit a PCA transform from samples y.

    Args:
        y:            [m, d] sample matrix.
        n_components: Number of principal components to keep (None = keep all).
        whiten:       If True, scale each PC by 1/sqrt(eigenvalue) so the
                      transformed data has identity covariance.  Full whitening
                      is fit_pca(y, n_components=None, whiten=True).

    Returns:
        A PCATransform that can be called on new samples or used to wrap h.
    """
    mu = y.mean(0)
    yc = y - mu
    cov = (yc.T @ yc) / max(1, y.shape[0] - 1)
    eps = 1e-8
    evals, evecs = torch.linalg.eigh(
        cov + eps * torch.eye(cov.shape[0], device=y.device, dtype=y.dtype)
    )
    # eigh returns ascending order; flip to descending
    evals = evals.flip(0)
    evecs = evecs.flip(1)
    if n_components is not None:
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]
    scales = (
        1.0 / torch.sqrt(evals.clamp_min(eps)) if whiten else torch.ones_like(evals)
    )
    return PCATransform(mu=mu, evecs=evecs, scales=scales)


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


Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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
    Linear kernel k(x, y) = x · y.

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

        k(x, y) = |x ∩ y| / |x ∪ y|  =  (x · y) / (|x| + |y| - x · y)

    This is a positive semi-definite kernel, making MMD^2 with it a valid
    divergence.  It is the standard similarity measure for Morgan fingerprints
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
    Useful for numerics, since some kernels have very small values
    """
    return lambda x, y: scale * kernel(x, y)


_KYY_MAX = 1024


def _mean_off_diagonal_kernel(
    y: torch.Tensor, kernel: Kernel, max_samples: int = _KYY_MAX
) -> torch.Tensor:
    if y.shape[0] < 2:
        raise ValueError("Need at least 2 target samples to compute MMD estimates.")

    if y.shape[0] <= max_samples:
        y_sub = y
    else:
        idx = torch.randperm(y.shape[0], device=y.device)[:max_samples]
        y_sub = y[idx]

    Kyy = kernel(y_sub, y_sub)
    m = y_sub.shape[0]
    return (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))


def auto_scale_kernel(
    model: Model[SampleType],
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    kernel: Kernel,
    *,
    batch_size: int,
    batches: int = 1,
    use_loo: bool = True,
    self_repulsion_weight: float = 1.0,
    target_constraint: float = 1.0,
    disable_pbar: bool = False,
) -> tuple[Kernel, float, float]:
    """
    Scale a kernel so the average initial constraint estimate over a few batches
    from the current model is target_constraint.

    Returns:
        (scaled_kernel, scale_factor, mean_initial_constraint_estimate)
    """
    if batches < 1:
        raise ValueError(f"batches must be at least 1, got {batches!r}")
    if self_repulsion_weight < 0:
        raise ValueError(
            "self_repulsion_weight must be nonnegative, "
            f"got {self_repulsion_weight!r}"
        )
    if target_constraint <= 0:
        raise ValueError(
            "target_constraint must be positive, " f"got {target_constraint!r}"
        )

    device = model.device
    dtype = torch.float32
    y_full = hstar.to(device=device, dtype=dtype)

    with torch.no_grad():
        kyy_mean = _mean_off_diagonal_kernel(y_full, kernel)
        estimates = []
        for _ in tqdm(
            range(batches),
            disable=disable_pbar,
            desc="Sampling for kernel auto-scaling",
        ):
            xs = model.sample(batch_size)
            x_feat = h(xs).detach().to(device=device, dtype=dtype)
            estimate = _mmd_calibration_terms(
                x_feat,
                y_full,
                kernel,
                kyy_mean=kyy_mean,
                use_loo=use_loo,
                self_repulsion_weight=self_repulsion_weight,
            ).estimate
            estimates.append(estimate)

    estimate_value = float(torch.stack(estimates).mean().item())

    eps = torch.finfo(dtype).eps
    if estimate_value <= eps:
        raise ValueError(
            "Initial constraint estimate must be positive to auto-scale the "
            f"kernel, got {estimate_value:.6g}."
        )

    scale = target_constraint / estimate_value
    return scale_kernel(kernel, scale), scale, estimate_value


@dataclass
class MMDCalibrationTerms:
    coeff: torch.Tensor
    estimate: torch.Tensor


def _mmd_calibration_terms(
    x: torch.Tensor,  # [n, d]
    y: torch.Tensor,  # [m, d]
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    kyy_mean: torch.Tensor,
    use_loo: bool,
    self_repulsion_weight: float = 1.0,
) -> MMDCalibrationTerms:
    """
    Return per-sample REINFORCE coefficients and the corresponding scalar
    discrepancy estimate for logging.

    This helper always uses the U-statistic form of the model-model kernel
    term. When use_loo is True, it subtracts the unbiased leave-one-out
    baseline from each per-sample coefficient.
    """
    n = x.shape[0]
    if n < 2:
        raise ValueError(
            "Need at least 2 model samples to compute U-statistic MMD coefficients."
        )
    if use_loo and n < 3:
        raise ValueError(
            "Need at least 3 model samples to compute LOO MMD coefficients."
        )

    Kxy = kernel(x, y)  # [n, m]
    Kxx = kernel(x, x)  # [n, n]
    cross_i = Kxy.mean(dim=1)  # [n]
    row_sum_excl_diag = Kxx.sum(dim=1) - Kxx.diag()  # [n]
    self_i = row_sum_excl_diag / (n - 1)  # [n]
    g_i = 2.0 * (self_repulsion_weight * self_i - cross_i)  # [n]

    if use_loo:
        self_excluding_m = (row_sum_excl_diag[:, None] - Kxx) / (n - 2)  # [n, n]
        g_excluding_m = 2.0 * (
            self_repulsion_weight * self_excluding_m - cross_i[:, None]
        )  # [n, n]
        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        loo_baseline_i = g_excluding_m.masked_fill(~offdiag_mask, 0.0).sum(dim=0) / (
            n - 1
        )
        g_i = g_i - loo_baseline_i

    estimate = self_repulsion_weight * self_i.mean() - 2.0 * cross_i.mean() + kyy_mean
    return MMDCalibrationTerms(coeff=g_i / n, estimate=estimate)


def _abs_stats(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x_abs = x.abs()
    return {
        f"{prefix}_abs_mean": x_abs.mean().item(),
        f"{prefix}_abs_max": x_abs.max().item(),
        f"{prefix}_abs_q95": torch.quantile(x_abs, 0.95).item(),
        f"{prefix}_abs_q99": torch.quantile(x_abs, 0.99).item(),
    }


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().pow(2).sum().item()
    return total**0.5


def calibrate_mmd(
    model: Model[SampleType],
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,  # target feature samples, shape [m, d]
    lambd: float,
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_weighting: Literal["raw", "normalized"] = "raw",
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    lr_scheduler_cls: Optional[Type[optim.lr_scheduler.LRScheduler]] = None,
    scheduler_params: Optional[dict[str, Any]] = None,
    cosine_schedule: bool = False,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.01,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = True,
    self_repulsion_weight: float = 1.0,
    grad_clip_norm: float | None = None,
    kernel_scale: float | Literal["auto"] | None = None,
    auto_scale_target_constraint: float = 10.0,
    auto_scale_batches: int = 5,
    logger: Callable[
        [dict[str, Any], Any, Model, Model], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: utils.CheckpointFn = None,
    disable_pbar: bool = False,
) -> Model:
    """
    Calibrate a generative model by minimizing MMD^2 (or energy distance) combined
    with a KL-to-base regularizer, in the same style as calibrate_relaxed.

    The per-sample coefficient is computed from the given kernel. Pass
    energy_distance_kernel() for energy distance or rbf_mixture_kernel() for
    MMD with a Gaussian kernel mixture.

    Preprocessing (whitening, PCA) should be applied to h and hstar before calling;
    see fit_pca() and PCATransform.

    Arguments mirror calibrate_relaxed; hstar is *samples* from the target for h.
    self_repulsion_weight controls the coefficient on the model-model kernel
    term; values other than 1.0 define a generalized, not strictly proper,
    discrepancy. kernel_scale can be a positive float or "auto"; the latter
    rescales the kernel so the average initial constraint estimate over
    auto_scale_batches batches from the base model is
    auto_scale_target_constraint. If grad_clip_norm is provided, gradients are
    clipped by global norm before each optimizer step.
    """
    if not use_loo:
        print("Warning: use_loo=False is only for comparison and does not work well.")
    if loss_weighting not in {"raw", "normalized"}:
        raise ValueError(
            "loss_weighting must be one of {'raw', 'normalized'}, "
            f"got {loss_weighting!r}"
        )
    if self_repulsion_weight < 0:
        raise ValueError(
            "self_repulsion_weight must be nonnegative, "
            f"got {self_repulsion_weight!r}"
        )

    device = model.device
    dtype = torch.float32

    y_full = hstar.to(device=device, dtype=dtype)
    base_model = utils.clone_network(model)
    if kernel_scale == "auto":
        kernel, kernel_scale_factor, kernel_scale_estimate = auto_scale_kernel(
            base_model,
            h,
            y_full,
            kernel,
            batch_size=batch_size,
            batches=auto_scale_batches,
            use_loo=use_loo,
            self_repulsion_weight=self_repulsion_weight,
            target_constraint=auto_scale_target_constraint,
            disable_pbar=disable_pbar,
        )
    elif kernel_scale is None:
        kernel_scale_factor = 1.0
        kernel_scale_estimate = None
    else:
        if kernel_scale <= 0:
            raise ValueError(
                "kernel_scale must be positive or 'auto', " f"got {kernel_scale!r}"
            )
        kernel = scale_kernel(kernel, float(kernel_scale))
        kernel_scale_factor = float(kernel_scale)
        kernel_scale_estimate = None

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    if cosine_schedule:
        if lr_scheduler_cls is not None:
            raise ValueError(
                "cosine_schedule=True is incompatible with lr_scheduler_cls. "
                "Set lr_scheduler_cls=None when using cosine_schedule."
            )
        scheduler = make_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        )
    else:
        scheduler_params = (
            {"T_max": epochs, "eta_min": 1e-6}
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

    with torch.no_grad():
        kyy_mean = _mean_off_diagonal_kernel(y_full, kernel)

    pbar = tqdm(range(epochs), desc="Training Epochs", disable=disable_pbar)
    for epoch in pbar:
        # ---- Sample from model (no grad) ----
        with torch.no_grad():
            xs = model.sample(batch_size)

        optimizer.zero_grad(set_to_none=True)

        # ---- KL-to-base coefficients ----
        with torch.no_grad():
            if hasattr(xs, "log_p_theta"):
                log_p_theta = xs.log_p_theta
            else:
                log_p_theta = log_p_chunked(
                    model, xs, batch_size, batch_chunks, samp_chunks
                )
        log_p_base = log_p_chunked(
            base_model, xs, batch_size, batch_chunks, samp_chunks
        )
        kls = log_p_theta - log_p_base
        kl_loss_val = kls.mean()
        kls -= (kls.sum() - kls) / (batch_size - 1)
        c_kl = kls / batch_size

        # ---- MMD / ED per-sample coefficients ----
        x_feat = h(xs).detach().to(device=device, dtype=dtype)

        with torch.no_grad():
            mmd_terms = _mmd_calibration_terms(
                x_feat,
                y_full,
                kernel,
                kyy_mean=kyy_mean,
                use_loo=use_loo,
                self_repulsion_weight=self_repulsion_weight,
            )
            c_ed = mmd_terms.coeff
            # ---- Full objective estimate for logging ----
            constraint_est = mmd_terms.estimate

        # ---- Mix coefficients ----
        if loss_weighting == "raw":
            kl_weight = lambd
            constraint_weight = 1.0
        else:
            denom = 1.0 + lambd
            kl_weight = lambd / denom
            constraint_weight = 1.0 / denom

        c_total = kl_weight * c_kl + constraint_weight * c_ed
        total_loss_val = kl_weight * kl_loss_val + constraint_weight * constraint_est

        # ---- Accumulate gradient via chunked log-prob increments ----
        for i in range(batch_chunks):
            min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
            c_i = c_total[min_idx:max_idx]
            for j in range(samp_chunks):
                delta_ij = model.log_p(
                    xs,
                    batch_idx=i,
                    batch_chunks=batch_chunks,
                    sample_idx=j,
                    sample_chunks=samp_chunks,
                )
                loss_ij = (c_i * delta_ij).sum()
                loss_ij.backward()

        grad_norm_before = _grad_norm(model.parameters())
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        grad_norm_after = _grad_norm(model.parameters())

        optimizer.step()
        scheduler.step() if scheduler is not None else None

        # ---- Logging / progress ----
        loss_item = float(total_loss_val.detach().cpu())
        pbar.set_postfix(
            {
                "loss": f"{loss_item:.4f}",
                "constraint": f"{float(constraint_est.cpu()):.4f}",
                "kl": f"{float(kl_loss_val.cpu()):.4f}",
            }
        )
        if logger is not None:
            with torch.no_grad():
                current_lr = optimizer.param_groups[0]["lr"]
                logger(
                    {
                        "epoch": epoch,
                        "lr": current_lr,
                        "loss": loss_item,
                        "constraint_loss": constraint_est.item(),
                        "kl_loss": kl_loss_val.item(),
                        "self_repulsion_weight": self_repulsion_weight,
                        "loss_weighting": loss_weighting,
                        "kl_weight": kl_weight,
                        "constraint_weight": constraint_weight,
                        "grad_norm": grad_norm_before,
                        "grad_norm_clipped": grad_norm_after,
                        "kernel_scale": kernel_scale_factor,
                        "kernel_scale_initial_constraint": kernel_scale_estimate,
                        **_abs_stats("c_kl", c_kl),
                        **_abs_stats("c_ed", c_ed),
                        **_abs_stats("c_total", c_total),
                    },
                    model,
                    base_model,
                    xs,
                )

        if checkpoint_fn is not None:
            checkpoint_fn(model, loss_item, optimizer, scheduler, epoch)

    return model
