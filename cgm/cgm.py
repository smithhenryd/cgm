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
                        "h_bar": hx.mean(0).detach().cpu(),
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
