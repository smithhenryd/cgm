from cgm.model import Model, SampleType
from cgm import utils

from typing import Any, Callable, Optional, Type, Union, TypeVar

import torch
import torch.optim as optim
from tqdm import tqdm


SampleType = TypeVar("SampleType")


def calibrate_relaxed(
    model: Model[SampleType],
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    lambd: float,
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    lr_scheduler_cls: Optional[
        Type[optim.lr_scheduler.LRScheduler]
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = True,
    logger: Callable[
        [dict[str, Any], Model, Model, SampleType], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: Optional[utils.CheckpointFn] = None,
    disable_pbar: bool = False,
) -> Model:
    """
    Calibrates a generative model according to the CGM-relax algorithm

    model: the base model to be calibrated
    h: the statistic to which, together with hstar, the base model is calibrated
    hstar: the expected value of h to which the base model is calibrated
    lambd: the regularization parameter
    epochs: number of CGM-relax iterations to perform
    batch_size: number of samples to draw per epoch
    optimizer_cls: torch optimizer class for updating the model parameters, default Adam
    optimizer_params: any parameters of the optimizer
    lr_scheduler_cls: torch learning rate scheduler class, default cosine
    scheduler_params: any parameters of the optimizer
    samp_chunks: number of splits across a single sample for computing gradients (eg., in a neural-SDE), default 1
    batch_chunks: number of splits across batches for computing gradients, default 1
    use_loo: a boolean, whether or not to use the leave-one-out gradient estimate, default True
    logger: a function for logging metrics during training
    checkpoint_fn: a function, called after each epoch to save model checkpoints during training
    disable_pbar: a boolean, setting True disables the tqdm progress bar
    """

    # SETUP
    ## Clone base model so gradients are not tracked
    ## NOTE: gradients are still tracked in model
    base_model = utils.clone_network(model)

    ## Optimizer
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    ## Learning rate scheduler
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

    # PERFORM CALIBRATION
    pbar = tqdm(range(epochs), desc="Training Epochs", disable=disable_pbar)
    for epoch in pbar:

        ## Draw samples from the model without gradients
        with torch.no_grad():
            xs = model.sample(batch_size)

        ## Compute gradients of the model
        optimizer.zero_grad(set_to_none=True)

        ### First compute log prob without tracking any gradients
        with torch.no_grad():
            if hasattr(xs, "log_p_theta"):
                log_p_theta = xs.log_p_theta
            else:
                log_p_theta = log_p_chunked(
                    model, xs, batch_size, batch_chunks, samp_chunks
                )  # (batch_size)
        log_p_base = log_p_chunked(
            base_model, xs, batch_size, batch_chunks, samp_chunks
        )  # (batch_size)
        kls = log_p_theta - log_p_base
        kl_loss_val = kls.mean()

        if use_loo:
            kls -= (kls.sum() - kls) / (batch_size - 1)
        hx = h(xs)

        #### Use a "dummy" weight to compute the coefficient on the gradient that multiplies each sample
        w_dummy = torch.ones(batch_size, device=model.device, requires_grad=True)
        viol_loss_dummy = utils.compute_violation_loss(hx, hstar, w_dummy)
        c_viol = torch.autograd.grad(viol_loss_dummy, w_dummy, retain_graph=False)[
            0
        ].detach()  # coefficient on violation loss
        viol_loss_val = viol_loss_dummy.detach()
        c_kl = kls / batch_size  # coefficient on KL

        c = lambd * c_kl + c_viol  # total coefficient
        total_loss_val = lambd * kl_loss_val + viol_loss_val  # total loss

        for i in range(batch_chunks):

            #### Extract relevant subset of samples
            min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
            c_i = c[min_idx:max_idx]
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

        ## Step the optimizer and lr scheduler
        optimizer.step()
        scheduler.step() if scheduler is not None else None

        loss_item = total_loss_val.item()
        pbar.set_postfix(
            {
                "loss": f"{loss_item:.4f}",
                "viol": f"{viol_loss_val.item():.4f}",
                "kl": f"{kl_loss_val.item():.4f}",
            }
        )

        ## Log the loss from epoch
        if logger is not None:
            with torch.no_grad():
                logger(
                    {
                        "epoch": epoch,
                        "loss": total_loss_val.item(),
                        "constraint_loss": viol_loss_val.item(),
                        "kl_loss": kl_loss_val.item(),
                        "h_bar": hx.mean(0).detach().cpu().item(),
                        "theta": torch.sigmoid(model.logit_p).item()
                        if hasattr(model, "logit_p") else float("nan"),
                    },
                    model,
                    base_model,
                    xs,
                )
        ## Checkpoint the model
        if checkpoint_fn is not None:
            checkpoint_fn(model, loss_item, optimizer, scheduler, epoch)

    # RETURN THE CALIBRATED MODEL
    return model

def calibrate_relaxed_offpolicy(
    model: Model[SampleType],
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    lambd: float,
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    proposal_optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    proposal_optimizer_params: dict[str, Any] = {},
    lr_scheduler_cls: Optional[
        Type[optim.lr_scheduler.LRScheduler]
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = True,
    logger: Callable[
        [dict[str, Any], Model, Model, Model, SampleType], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: Optional[utils.CheckpointFn] = None,
    disable_pbar: bool = False,
) -> Model:

    # === SETUP ======================================================
    base_model = utils.clone_network(model)
    proposal_model = utils.clone_network(model, disable_gradients=False)       # (3) NEW

    optimizer_theta = optimizer_cls(model.parameters(), **optimizer_params)
    if 'lr' not in proposal_optimizer_params:
        proposal_optimizer_params['lr'] = 10 * optimizer_params.get('lr', 1e-3)
    optimizer_phi = proposal_optimizer_cls(
        proposal_model.parameters(), **proposal_optimizer_params
    )

    scheduler_params = (
        {"T_max": epochs, "eta_min": 1e-6}
        if (scheduler_params is None and lr_scheduler_cls is optim.lr_scheduler.CosineAnnealingLR)
        else scheduler_params
    )
    scheduler_theta = (
        lr_scheduler_cls(optimizer_theta, **scheduler_params)
        if lr_scheduler_cls is not None
        else None
    )

    pbar = tqdm(range(epochs), desc="Training Epochs (off-policy)", disable=disable_pbar)

    # === LOOP ========================================================
    for epoch in pbar:

        # --------------------------------------------------------------
        # 1. SAMPLE FROM PROPOSAL q_phi (off-policy)
        # --------------------------------------------------------------
        with torch.no_grad():
            xs = proposal_model.sample(batch_size)

        # --------------------------------------------------------------
        # 2. COMPUTE log_p_theta (NO GRAD) and log_q_phi (WITH GRAD)
        # --------------------------------------------------------------
        # log_p_theta does NOT require grad
        with torch.no_grad():
            log_p_theta = log_p_chunked(
                model, xs, batch_size, batch_chunks, samp_chunks
            )

        log_q_phi = log_p_chunked(
            proposal_model, xs, batch_size, batch_chunks, samp_chunks
        )

        # importance weights w = p_theta / q_phi (detached)
        w = torch.exp((log_p_theta - log_q_phi).detach())

        # --------------------------------------------------------------
        # 3. CONSTRAINT VIOLATION
        # --------------------------------------------------------------
        hx = h(xs)

        # dummy weights to extract per-sample coefficients c_viol
        w_dummy = torch.ones(batch_size, device=model.device, requires_grad=True)
        viol_loss_dummy = utils.compute_violation_loss(hx, hstar, w_dummy * w)
        c_viol = torch.autograd.grad(viol_loss_dummy, w_dummy)[0].detach()
        viol_loss_val = viol_loss_dummy.detach()

        # --------------------------------------------------------------
        # 4. KL TERM (same as original, but using xs ~ proposal)
        # --------------------------------------------------------------
        with torch.no_grad():
            log_p_base = log_p_chunked(
                base_model, xs, batch_size, batch_chunks, samp_chunks
            )
        kls = log_p_theta - log_p_base
        if use_loo:
            kls = kls - (kls.sum() - kls) / (batch_size - 1)
        c_kl = kls / batch_size
        kl_loss_val = kls.mean()

        # --------------------------------------------------------------
        # 5. TOTAL PER-SAMPLE COEFFICIENTS
        # --------------------------------------------------------------
        c = lambd * c_kl + c_viol
        total_loss_val = viol_loss_val + lambd * kl_loss_val

        # --------------------------------------------------------------
        # 6 & 7. JOINT UPDATE OF θ AND φ USING PER-SAMPLE GRADIENTS
        # --------------------------------------------------------------
        optimizer_theta.zero_grad(set_to_none=True)
        psi_norms = []

        for m in range(batch_size):
            # scalar weight for this sample

            # ----------------------------------------------------------
            # θ update contribution from sample m:
            #   loss_theta_m = c[m] * log p_theta(x_m)
            # ----------------------------------------------------------
            logp_single = model.log_p(xs[m:m+1])          # scalar tensor
            loss_theta_m = c[m] * logp_single

            # gradient wrt θ for this sample (used for θ and ψ)
            grads_m = torch.autograd.grad(
                loss_theta_m,
                model.parameters(),
                retain_graph=True,
                allow_unused=True,
                create_graph=False,   # we don't need higher-order grads
            )

            # Accumulate θ gradients manually
            for p, g in zip(model.parameters(), grads_m):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.add_(g.detach())

            # ----------------------------------------------------------
            # ψ_m = w[m] * c[m] * ∇_θ log p_theta(x_m)
            psi_m = w[m] * torch.stack(grads_m).detach()
            psi_norms.append((psi_m * psi_m).sum())

        # apply θ update
        optimizer_theta.step()
        if scheduler_theta is not None:
            scheduler_theta.step()

        psi_norms = torch.stack(psi_norms)  # [batch_size]

        # --------------------------------------------------------------
        # φ UPDATE: minimize E_q [ ||ψ||^2 * log q_phi ]
        # --------------------------------------------------------------
        loss_phi = -(psi_norms.detach() * log_q_phi).mean()

        optimizer_phi.zero_grad(set_to_none=True)
        loss_phi.backward()
        optimizer_phi.step()

        # --------------------------------------------------------------
        # 8. LOGGING
        # --------------------------------------------------------------
        pbar.set_postfix({
            "loss": f"{total_loss_val.item():.4f}",
            "viol": f"{viol_loss_val.item():.4f}",
            "kl": f"{kl_loss_val.item():.4f}",
        })

        if logger is not None:
            with torch.no_grad():
                logger(
                    {
                        "epoch": epoch,
                        "loss": total_loss_val.item(),
                        "constraint_loss": viol_loss_val.item(),
                        "kl_loss": kl_loss_val.item(),
                        # h_bar but weighted by importance weights
                        "h_bar": (w[:, None] * hx).mean(0).detach().cpu().item(),
                        "proposal_loss": loss_phi.item(),
                        "phi": proposal_model.model_p().item()
                        if hasattr(proposal_model, "logit_p") else float("nan"),
                        "theta": model.model_p().item()
                        if hasattr(model, "logit_p") else float("nan"),
                    },
                    model,
                    base_model,
                    proposal_model,
                    xs,
                )

        if checkpoint_fn is not None:
            checkpoint_fn(model, total_loss_val.item(), optimizer_theta, scheduler_theta, epoch)

    return model




def _compute_sr_weights(hstar_val: float, batch_size: int) -> tuple[float, float]:
    """
    Precompute (w0, w1) for the Stationary REINFORCE gradient estimator.

    Solves for weights satisfying:
      (i)  w(1) - w(0) = 1   (normalization; matches absolute-error scaling far from h*)
      (ii) E_{h*}[U_hat] = 0  (stationarity; zero expected update at p = h*)

    The solution is:
      w(0) = -(1-h*) * w0_factor / D
      w(1) =  h*     * w1_factor / D,   D = 2*alpha*(h* - beta)
    where  w0_factor = h* - 2*alpha*beta       (> 0 always, since beta < h*)
           w1_factor = 2*alpha*(1-beta) - (1-h*)  (> 0 when M is not too large relative to 1/h*)
    """
    M = batch_size
    h = hstar_val

    if h <= 0.0 or h >= 1.0:
        return -(1.0 - h), h

    from torch.distributions import Binomial as _Binomial

    k_vals = torch.arange(M + 1, dtype=torch.float64)
    dist = _Binomial(total_count=M, probs=torch.tensor(h, dtype=torch.float64))
    pmf = dist.log_prob(k_vals).exp()  # shape (M+1,)

    mask = k_vals < (M * h)
    alpha = pmf[mask].sum().item()
    assert alpha > 1e-15, "Unstable if alpha < 1e-15"

    beta = ((k_vals[mask] / M) * pmf[mask]).sum().item() / alpha

    w0_factor = h - 2.0 * alpha * beta                    # h* - 2αβ; > 0 always
    w1_factor = 2.0 * alpha * (1.0 - beta) - (1.0 - h)   # 2α(1-β) - (1-h*); > 0 for M not too large

    D = 2.0 * alpha * (h - beta)
    w0 = -(1.0 - h) * w0_factor / D
    w1 = h * w1_factor / D
    return float(w0), float(w1)


def calibrate_stationary_reinforce(
    model: Model[SampleType],
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    lambd: float = 0.0,
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: Type[optim.Optimizer] = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    lr_scheduler_cls: Optional[
        Type[optim.lr_scheduler.LRScheduler]
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = True,
    logger: Callable[
        [dict[str, Any], Model, Model, SampleType], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: Optional[utils.CheckpointFn] = None,
    disable_pbar: bool = False,
) -> Model:
    """
    Calibrates a generative model using the Stationary REINFORCE gradient estimator.

    Implements the stationary estimator from stationary_reinforce.tex:
      U_hat = [1/M * sum_m nabla log p_theta(x_m) * w(h(x_m))] * (-1)^I[y < h*]
    where w(0) and w(1) are chosen so E[U_hat] = 0 when E_theta[h(x)] = h* and
    gradient magnitudes are symmetric under h* <-> 1-h*.

    h must be binary: h(x) in {0, 1}.
    """
    hstar_list = hstar.cpu().tolist()
    if not isinstance(hstar_list, list):
        hstar_list = [hstar_list]
    n_constraints = len(hstar_list)

    # Precompute (w0, w1) for each constraint based on h* and batch_size
    weights = [_compute_sr_weights(hv, batch_size) for hv in hstar_list]
    w0_vals = torch.tensor([w[0] for w in weights], dtype=torch.float32, device=model.device)
    w1_vals = torch.tensor([w[1] for w in weights], dtype=torch.float32, device=model.device)

    # Clone base model only if KL regularization is used
    base_model = utils.clone_network(model) if lambd > 0.0 else model

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

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

    pbar = tqdm(range(epochs), desc="Training Epochs (SR)", disable=disable_pbar)
    for epoch in pbar:

        with torch.no_grad():
            xs = model.sample(batch_size)

        # hx: (batch_size, n_constraints), expected binary {0, 1}
        hx = h(xs)
        if hx.dim() == 1:
            hx = hx.unsqueeze(-1)

        # Batch fraction y_j and sign for each constraint
        y = hx.mean(dim=0)  # (n_constraints,)
        sign = (1.0 - 2.0 * (y < hstar.to(model.device)).float())  # (n_constraints,)

        # Per-sample per-constraint weights: w(h=0)=w0_j, w(h=1)=w1_j
        w_m = torch.where(hx > 0.5, w1_vals.unsqueeze(0), w0_vals.unsqueeze(0).expand(batch_size, -1))

        # Sum over constraints; divide by M to get per-sample coefficient c
        c_viol = (sign.unsqueeze(0) * w_m).sum(dim=1) / batch_size  # (batch_size,)

        # Optional KL regularization
        kl_loss_val = torch.tensor(0.0, device=model.device)
        if lambd > 0.0:
            with torch.no_grad():
                log_p_theta = log_p_chunked(model, xs, batch_size, batch_chunks, samp_chunks)
                log_p_base = log_p_chunked(base_model, xs, batch_size, batch_chunks, samp_chunks)
            kls = log_p_theta - log_p_base
            kl_loss_val = kls.mean()
            if use_loo:
                kls = kls - (kls.sum() - kls) / (batch_size - 1)
            c = lambd * kls / batch_size + c_viol
        else:
            c = c_viol

        # Violation loss for logging (no grad)
        with torch.no_grad():
            viol_loss_val = utils.compute_violation_loss(
                hx, hstar.to(model.device), torch.ones(batch_size, device=model.device)
            )

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        for i in range(batch_chunks):
            min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
            c_i = c[min_idx:max_idx]
            for j in range(samp_chunks):
                delta_ij = model.log_p(
                    xs,
                    batch_idx=i,
                    batch_chunks=batch_chunks,
                    sample_idx=j,
                    sample_chunks=samp_chunks,
                )
                (c_i * delta_ij).sum().backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_val = viol_loss_val + lambd * kl_loss_val

        pbar.set_postfix({
            "loss": f"{total_loss_val.item():.4f}",
            "viol": f"{viol_loss_val.item():.4f}",
            "kl": f"{kl_loss_val.item():.4f}",
        })

        if logger is not None:
            with torch.no_grad():
                logger(
                    {
                        "epoch": epoch,
                        "loss": total_loss_val.item(),
                        "constraint_loss": viol_loss_val.item(),
                        "kl_loss": kl_loss_val.item(),
                        "h_bar": hx.mean(0).detach().cpu().squeeze().item()
                        if n_constraints == 1 else float("nan"),
                        "theta": torch.sigmoid(model.logit_p).item()
                        if hasattr(model, "logit_p") else float("nan"),
                    },
                    model,
                    base_model,
                    xs,
                )

        if checkpoint_fn is not None:
            checkpoint_fn(model, total_loss_val.item(), optimizer, scheduler, epoch)

    return model


def calibrate_reward(
    model: Model,
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    N_samp: int,
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: optim.Optimizer = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    lr_scheduler_cls: Optional[
        optim.lr_scheduler.LRScheduler
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = True,
    logger: Callable[
        [dict[str, Any], Model, Model, SampleType], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: Optional[utils.CheckpointFn] = None,
    disable_pbar: bool = False,
    dual_max_iters: int = 2500,
) -> Model:
    """
    Calibrates a generative model according to the CGM-reward algorithm

    N_samp: the number of samples with which to estimate the parameters alpha* of the maximum entropy distribution
    """

    # SETUP
    ## Clone base model so gradients are not tracked
    ## NOTE: gradients are still tracked in model
    base_model = utils.clone_network(model)

    ## Optimizer
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    ## Learning rate scheduler
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

    # ESTIMATE ALPHA*
    with torch.no_grad():
        xs = model.sample(N_samp)
    alpha_hat = utils.solve_dual(h(xs), hstar, max_iters=dual_max_iters)

    # PERFORM CALIBRATION
    pbar = tqdm(range(epochs), desc="Training Epochs", disable=disable_pbar)
    for epoch in pbar:

        ## Draw samples from the model without gradients
        with torch.no_grad():
            xs = model.sample(batch_size)

        with torch.no_grad():
            if hasattr(xs, "log_p_theta"):
                log_p_theta = xs.log_p_theta
            else:
                log_p_theta = log_p_chunked(
                    model, xs, batch_size, batch_chunks, samp_chunks
                )  # (batch_size)

        ## KL term
        log_p_base = log_p_chunked(
            base_model, xs, batch_size, batch_chunks, samp_chunks
        )  # (batch_size)
        kls = log_p_theta.detach() - log_p_base

        ## Reward term
        hx = h(xs)
        rs = torch.sum((-1) * alpha_hat[None, :] * hx, dim=-1)  # (batch_size)
        ys = kls + rs

        total_loss_val = ys.mean()

        if use_loo:
            ys -= (ys.sum() - ys) / (batch_size - 1)  # subtract off LOO baseline
        ys *= 1 / batch_size

        ## Gradient computation
        optimizer.zero_grad(set_to_none=True)

        ### Compute gradient in chunks
        for i in range(batch_chunks):

            #### Extract relevant subset of samples
            min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
            ys_i = ys[min_idx:max_idx]
            for j in range(samp_chunks):
                log_p_theta_ij = model.log_p(
                    xs,
                    batch_idx=i,
                    batch_chunks=batch_chunks,
                    sample_idx=j,
                    sample_chunks=samp_chunks,
                )
                weights_ij = torch.exp(log_p_theta_ij - log_p_theta_ij.detach())
                loss_ij = (ys_i * weights_ij).sum()
                loss_ij.backward()

        ## For logging, also compute the violation loss
        w_dummy = torch.ones(batch_size, device=model.device)
        viol_loss = utils.compute_violation_loss(
            hx, hstar, w_dummy
        )  # notice no gradients are tracked here

        viol_loss_val, kl_loss_val = (
            viol_loss,
            kls.mean(),
        )

        ## Step the optimizer and lr scheduler
        optimizer.step()
        scheduler.step() if scheduler is not None else None

        loss_item = total_loss_val.item()
        pbar.set_postfix(
            {
                "loss": f"{loss_item:.4f}",
                "viol": f"{viol_loss_val.item():.4f}",
                "kl": f"{kl_loss_val.item():.4f}",
            }
        )

        ## Log the loss from epoch
        if logger is not None:
            with torch.no_grad():
                logger(
                    {
                        "epoch": epoch,
                        "loss": total_loss_val.item(),
                        "constraint_loss": viol_loss_val.item(),
                        "kl_loss": kl_loss_val.item(),
                        "h_bar": hx.mean(0).detach().cpu(),
                    },
                    model,
                    base_model,
                    xs,
                )
        ## Checkpoint the model
        if checkpoint_fn is not None:
            checkpoint_fn(model, loss, optimizer, scheduler, epoch)

    # RETURN THE CALIBRATED MODEL
    return model


def calibrate_forward_kl(
    model: Model,
    h: Callable[[SampleType], torch.Tensor],
    hstar: torch.Tensor,
    N_samp: int,
    epochs: int = 1000,
    batch_size: int = 100,
    optimizer_cls: optim.Optimizer = optim.Adam,
    optimizer_params: dict[str, Any] = {"lr": 1e-3},
    lr_scheduler_cls: Optional[
        optim.lr_scheduler.LRScheduler
    ] = optim.lr_scheduler.CosineAnnealingLR,
    scheduler_params: Optional[dict[str, Any]] = None,
    samp_chunks: int = 1,
    batch_chunks: int = 1,
    use_loo: bool = False,
    logger: Callable[
        [dict[str, Any], Model, Model, SampleType], None
    ] = lambda x, *args: utils.default_logger(x),
    checkpoint_fn: Optional[utils.CheckpointFn] = None,
    disable_pbar: bool = False,
    dual_max_iters: int = 2500,
) -> Model:
    """
    Calibrates a generative model according to the Khalifa 2021 baseline

    N_samp: the number of samples with which to estimate the parameters alpha* of the maximum entropy distribution
    """

    # SETUP
    ## Clone base model so gradients are not tracked
    ## NOTE: gradients are still tracked in model
    base_model = utils.clone_network(model)

    ## Optimizer
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    ## Learning rate scheduler
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

    # ESTIMATE ALPHA*
    with torch.no_grad():
        xs = model.sample(N_samp)
    alpha_hat = utils.solve_dual(h(xs), hstar, max_iters=dual_max_iters)

    # PERFORM CALIBRATION
    pbar = tqdm(range(epochs), desc="Training Epochs", disable=disable_pbar)
    for epoch in pbar:

        ## Draw samples from the model without gradients
        with torch.no_grad():
            xs = model.sample(batch_size)

        with torch.no_grad():
            if hasattr(xs, "log_p_theta"):
                log_p_theta = xs.log_p_theta
            else:
                log_p_theta = log_p_chunked(
                    model, xs, batch_size, batch_chunks, samp_chunks
                )  # (batch_size)
            log_p_base = log_p_chunked(
                base_model, xs, batch_size, batch_chunks, samp_chunks
            )  # (batch_size)

        ## Reward tilt of pre-trained model
        hx = h(xs)
        rs = torch.sum(alpha_hat[None, :] * hx, dim=-1)  # (batch_size)
        log_p_star = rs + log_p_base

        ## Density ratio
        log_ratio = log_p_star - log_p_theta
        ratio = log_ratio.exp()
        total_loss_val = ratio.mean()

        if use_loo:
            ratio -= (ratio.sum() - ratio) / (
                batch_size - 1
            )  # subtract off LOO baseline
        ratio *= 1 / batch_size

        ## Gradient computation
        optimizer.zero_grad(set_to_none=True)

        ### Compute gradient in chunks
        for i in range(batch_chunks):

            #### Extract relevant subset of samples
            min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
            ratio_i = ratio[min_idx:max_idx]
            for j in range(samp_chunks):
                log_p_theta_ij = model.log_p(
                    xs,
                    batch_idx=i,
                    batch_chunks=batch_chunks,
                    sample_idx=j,
                    sample_chunks=samp_chunks,
                )
                loss_ij = -(ratio_i * log_p_theta_ij).sum()
                loss_ij.backward()

        ## For logging, also compute the violation loss
        w_dummy = torch.ones(batch_size, device=model.device)
        viol_loss = utils.compute_violation_loss(
            hx, hstar, w_dummy
        )  # notice no gradients are tracked here

        # KL for logging only
        kls = log_p_theta - log_p_base

        viol_loss_val, kl_loss_val = (
            viol_loss,
            kls.mean(),
        )

        ## Step the optimizer and lr scheduler
        optimizer.step()
        scheduler.step() if scheduler is not None else None

        loss_item = total_loss_val.item()
        pbar.set_postfix(
            {
                "loss": f"{loss_item:.4f}",
                "viol": f"{viol_loss_val.item():.4f}",
                "kl": f"{kl_loss_val.item():.4f}",
            }
        )

        ## Log the loss from epoch
        if logger is not None:
            with torch.no_grad():
                logger(
                    {
                        "epoch": epoch,
                        "loss": total_loss_val.item(),
                        "constraint_loss": viol_loss_val.item(),
                        "kl_loss": kl_loss_val.item(),
                        "h_bar": hx.mean(0).detach().cpu(),
                    },
                    model,
                    base_model,
                    xs,
                )
        ## Checkpoint the model
        if checkpoint_fn is not None:
            checkpoint_fn(model, loss, optimizer, scheduler, epoch)

    return model


def log_p_chunked(
    model: Model,
    xs: SampleType,
    batch_size: int,
    batch_chunks: int = 1,
    samp_chunks: int = 1,
) -> torch.Tensor:
    """
    Helper function for computing the log density of a generative model in chunks
    """
    out = None

    for i in range(batch_chunks):
        min_idx, max_idx = utils.chunk_bounds(batch_size, batch_chunks, i)
        acc = None  # accumulates log probabilities within a batch
        for j in range(samp_chunks):
            lp_ij = model.log_p(
                xs,
                batch_idx=i,
                batch_chunks=batch_chunks,
                sample_idx=j,
                sample_chunks=samp_chunks,
            )
            acc = lp_ij if acc is None else (acc + lp_ij)

            if out is None:  # allocate once we know dtype/device from model.log_p
                out = torch.empty(batch_size, device=lp_ij.device, dtype=lp_ij.dtype)
        out[min_idx:max_idx] = acc
    return out
