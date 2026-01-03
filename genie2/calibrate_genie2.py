import wandb  # use wandb for logging runs

from typing import Any, Tuple

import torch
import pandas as pd
import sys
import os
import argparse
from functools import partial

from genie_score_network import GenieScoreNetwork, drift, diffusion

import secstruct
from cgm.cgm import calibrate_relaxed, calibrate_reward
from cgm.utils import chunk_bounds, CheckpointEveryN

sys.path.append("../neural_sde")
from utils import BackwardDrift, compute_KL_est, save_checkpoint
from neural_sde import NeuralSDE


def wandb_logger(
    logs: dict[str, Any],
    model: NeuralSDE,
    base_model: NeuralSDE,
    x: dict[str, torch.Tensor],
    batch_chunks: int,
    samp_chunks: int,
) -> None:
    """
    Helper function for logging metrics using wandb
    """

    batch_size, T, _ = x.xs.shape
    T -= 1
    kls = torch.zeros(batch_size, device=x.xs.device, dtype=x.xs.dtype) # compute reduced variance KL estimate
    for i in range(batch_chunks):
        min_i, max_i = chunk_bounds(batch_size, batch_chunks, i)
        for j in range(samp_chunks):
            x_sub = model._extract_chunk(x, i, batch_chunks, j, samp_chunks)
            kls[min_i:max_i] += compute_KL_est(base_model, x_sub)
    kl_est = kls.mean()

    logs = {
        "regularization_loss": kl_est,
        "moment_loss": logs["constraint_loss"],
        "loss": logs["loss"],
    }
    wandb.log(logs)


def clean_df(csv_path: str) -> pd.DataFrame:
    """
    Reads in a df containing alpha helix and beta strand proportions and performs basic cleaning
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"% Helix": "alpha_helix", "% Strand": "beta_strand"})

    # Clean the data
    df = (
        df[["alpha_helix", "beta_strand"]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    df = df[
        df["alpha_helix"].between(0, 1) & df["beta_strand"].between(0, 1)
    ]  # proportion of residues in alpha helix and beta strand must be between 0. and 1.
    df = df[
        df["alpha_helix"] + df["beta_strand"] <= 1.0
    ]  # sum of alpha helix and beta strand proportions must be <= 1.

    return df


def compute_univariate_quantiles(
    df: pd.DataFrame, N_quantiles: int
) -> Tuple[dict[str:list], dict[str:list]]:

    # Quantiles of interest
    quantiles = torch.linspace(0.0, 1.0, N_quantiles + 2)[1:-1].tolist()
    alpha_q = df["alpha_helix"].quantile(quantiles, interpolation="linear")
    beta_q = df["beta_strand"].quantile(quantiles, interpolation="linear")
    quantiles = {"alpha": quantiles, "beta": quantiles}
    quantile_vals = {"alpha": alpha_q.tolist(), "beta": beta_q.tolist()}
    return quantiles, quantile_vals


def compute_bivariate_quantiles(
    df: pd.DataFrame, N_quantiles: int
) -> Tuple[list, dict[str:list]]:

    helix_prop = []
    strand_prop = []
    quantiles = []

    helix_ths = torch.linspace(0.0, 1.0, N_quantiles + 2)[1:].tolist()  # exclude 0
    strand_ths = torch.linspace(0.0, 1.0, N_quantiles + 2)[1:].tolist()  # exclude 0
    rows = []
    for h in helix_ths:
        for s in strand_ths:
            if (
                h == 1.0 and s == 1.0
            ):
                continue
            else:
                helix_prop.append(h)
                strand_prop.append(s)

            mask = (df["alpha_helix"] <= h) & (df["beta_strand"] <= s)
            quantiles.append(mask.mean().item())

    quantile_vals = {"alpha": helix_prop, "beta": strand_prop}
    return quantiles, quantile_vals


def h_uni(x: dict[str, torch.Tensor], quantile_vals: dict[str:float]) -> torch.Tensor:
    """
    Calibration function corresponding to univariate quantile constraints
    """
    x = x.xs[:, -1]

    hh_vals = []
    for b in range(x.shape[0]):
        x_b = x[b].reshape([-1, 3]).cpu().numpy()
        ss_val = secstruct.compute_quantiles(
            x_b, quantile_vals
        )  # determine whether or not % of helices, strands are less than the specified quantiles
        hh_vals.append(
            torch.cat(
                [
                    torch.tensor(ss_val[key], device=x.device, dtype=x.dtype)
                    for key, value in quantile_vals.items()
                ]
            ).reshape(1, -1)
        )
    hh_vals = torch.concat(hh_vals, axis=0)
    return hh_vals


def h_bi(x: dict[str, torch.Tensor], quantile_vals: dict[str:float]) -> torch.Tensor:
    """
    Calibration function corresponding to bivariate quantile constraints
    """
    x = x.xs[:, -1]

    hh_vals = []
    for b in range(x.shape[0]):
        x_b = x[b].reshape([-1, 3]).cpu().numpy()
        ss_val = secstruct.compute_joint_quantiles(
            x_b, quantile_vals
        )  # determine whether or not % of helices, % of strands are jointly less than the specified quantiles
        hh_vals.append(
            torch.tensor(ss_val, device=x.device, dtype=x.dtype).reshape(1, -1)
        )
    hh_vals = torch.concat(hh_vals, axis=0)
    return hh_vals

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str
    )  # path to file containing SS statistics of CATH domains
    parser.add_argument(
        "--calibration_mode", choices=["relax", "reward"]
    )  # 'relax' for CGM-relax, 'reward' for CGM-reward
    parser.add_argument("--epochs", type=int, default=100)  # number of CGM epochs
    parser.add_argument(
        "--samp_chunks", type=int, default=20
    )  # number of chunks within sample
    parser.add_argument(
        "--batch_chunks", type=int, default=4
    )  # number of chunks within batch
    parser.add_argument("--lr_init", type=float, default=1e-5) # initial learning rate
    parser.add_argument("--lr_min", type=float, default=1e-7)  # minimum learning rate
    parser.add_argument("--batch_size", type=int, default=64)  # batch_size
    parser.add_argument(
        "--lambda", dest="lambd", type=float, default=None
    )  # regularization parameter
    parser.add_argument(
        "--N", type=int, default=None
    )  # number of samples for CGM-reward
    parser.add_argument(
        "--noise_scale", type=float, default=0.5
    )  # noise scale for Genie2 base model
    parser.add_argument("--L", type=int, default=100)  # number of residues
    parser.add_argument(
        "--const_type", type=str, choices=["uni", "bi"], default="bi"
    )  # type of constraint (univariate 'uni' or bivariate 'bi')
    parser.add_argument(
        "--N_quantiles", type=int, default=3
    )  # number of quantile constraints to consider
    parser.add_argument(
        "--N_samples", type=int, default=10**3
    )  # number of pdb files to save
    parser.add_argument(
        "--dir", dest="output_dir", type=str, default=None
    )  # directory for saving pdb files
    parser.add_argument(
        "--ckpt_dir", type=str, default=None
    )  # directory for saving model checkpoint files
    parser.add_argument(
        "--ckpt_every", type=int, default=None
    )  # checkpoint the Genie2 model every ckpt_every epochs

    args = parser.parse_args()
    (
        path,
        alg,
        epochs,
        samp_chunks,
        batch_chunks,
        lr_init,
        lr_min,
        batch_size,
        lambd,
        N,
        noise_scale,
        L,
        const_type,
        N_quantiles,
        N_samples,
        output_dir,
        ckpt_dir,
        ckpt_every,
    ) = (
        args.path,
        args.calibration_mode,
        args.epochs,
        args.samp_chunks,
        args.batch_chunks,
        args.lr_init,
        args.lr_min,
        args.batch_size,
        args.lambd,
        args.N,
        args.noise_scale,
        args.L,
        args.const_type,
        args.N_quantiles,
        args.N_samples,
        args.output_dir,
        args.ckpt_dir,
        args.ckpt_every,
    )

    # Directory for saving pdb files
    output_str = alg + "_" + const_type + "_" + str(N_quantiles) + "_"
    if output_dir is None:
        output_dir = (
            (output_str + str(lambd).replace(".", "_"))
            if alg == "relax"
            else (output_str + str(N))
        )

    #  Directory for saving model checkpoints
    if (ckpt_every is not None) and (ckpt_every <= epochs):
        if ckpt_dir is None:
            ckpt_dir = (
                (output_str + str(lambd).replace(".", "_"))
                if alg == "relax"
                else (output_str + str(N))
            )
        ckpt_dir = os.path.join("checkpoints", ckpt_dir)
        os.makedirs(
            ckpt_dir, exist_ok=True
        )  # make the checkpoint directory, if it doesn't already exist
        ckpt_fn = CheckpointEveryN(ckpt_dir=ckpt_dir, N=ckpt_every)
    else:
        ckpt_fn = None

    # Perform calibration on a single GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine the calibration function
    df = clean_df(path)  # read in data
    if const_type == "uni":  # univariate calibration constraint
        quantiles, quantile_vals = compute_univariate_quantiles(df, N_quantiles)
        hstar = torch.cat(
            [torch.tensor(value, device=device) for key, value in quantiles.items()]
        )
        hh = lambda x: h_uni(x, quantile_vals)
    else:  # bivariate calibration constraint
        quantiles, quantile_vals = compute_bivariate_quantiles(df, N_quantiles)
        hstar = torch.tensor(quantiles, device=device)
        hh = lambda x: h_bi(x, quantile_vals)

    # Time grid for discretizing sampling from the diffusion model
    N_steps = 100
    mid_pt = 0.05
    t_grid_1 = torch.linspace(0.0, mid_pt, N_steps // 2, device=device)
    t_grid_2 = torch.linspace(
        mid_pt + t_grid_1[-1] - t_grid_1[-2],
        1.0,
        (N_steps + 1) - N_steps // 2,
        device=device,
    )
    t_grid = torch.concat([t_grid_1, t_grid_2])

    # Instantiate the base Genie model
    genie_score = GenieScoreNetwork(device, noise_scale, None)
    sqrt_alphas, sqrt_betas = (
        genie_score.denoiser.sqrt_alphas,
        genie_score.denoiser.sqrt_betas,
    )
    f = lambda x, t: drift(x, t, sqrt_alphas)
    sigma = lambda t: diffusion(t, sqrt_betas, noise_scale)
    genie_drift_network = BackwardDrift(genie_score, f, sigma)  # the base Genie2 drift
    base_model = NeuralSDE(
        sde_dim=L * 3,
        drift=genie_drift_network,
        diffusion=lambda t: sigma(1.0 - t),
        t_grid=t_grid,
    )  # the base Genie2 neural-SDE

    # Initialize wandb run
    run = wandb.init(
        project="genie-finetune",
        config={
            "noise_scale": noise_scale,
            "num_residues": L,
            "batch_size": batch_size,
            "num_epochs": epochs,
            "Nsteps": N_steps,
            "lr_init": lr_init,
            "lr_min": lr_min,
            "num_quantiles": N_quantiles,
        },
    )

    if alg == "relax":  # calibrate Genie2 with CGM-relax
        wandb.config.update({"lambda": lambd}, allow_val_change=True)
        calibrated_model = calibrate_relaxed(
            base_model,
            hh,
            hstar,
            lambd,
            epochs,
            batch_size,
            optimizer_params={"lr": lr_init},
            scheduler_params={"T_max": epochs, "eta_min": lr_min},
            samp_chunks=samp_chunks,
            batch_chunks=batch_chunks,
            use_loo=True,
            logger=partial(
                wandb_logger,
                batch_chunks=batch_chunks,
                samp_chunks=samp_chunks,
            ),
            checkpoint_fn=ckpt_fn,
            disable_pbar=True,
        )
    else:  # calibrate Genie2 with CGM-reward
        wandb.config.update({"N_samples": N}, allow_val_change=True)
        calibrated_model = calibrate_reward(
            base_model,
            hh,
            hstar,
            N,
            epochs,
            batch_size,
            optimizer_params={"lr": lr_init},
            scheduler_params={"T_max": epochs, "eta_min": lr_min},
            samp_chunks=samp_chunks,
            batch_chunks=batch_chunks,
            use_loo=True,
            logger=partial(
                wandb_logger,
                batch_chunks=batch_chunks,
                samp_chunks=samp_chunks,
            ),
            checkpoint_fn=ckpt_fn,
            disable_pbar=True,
        )

    # Once we have performed calibration, sample structures
    if N_samples > 0:
        with torch.no_grad():
            samps = calibrated_model.sample(N_samples)
            samps = samps.xs.cpu().detach().numpy()[:, -1, :]

        out_dir = os.path.join("genie_outputs", output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for n in range(samps.shape[0]):
            x0 = samps[n]
            x0 = x0.reshape(x0.shape[0] // 3, 3)
            secstruct.save_pdb(x0, os.path.join(out_dir, f"sample_{n}.pdb"))


if __name__ == "__main__":
    main()