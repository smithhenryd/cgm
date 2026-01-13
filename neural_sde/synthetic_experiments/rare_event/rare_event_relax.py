"""
Script for upweighting a rare mode in a GMM with CGM-relax
Reproduces Figure 1B in the paper
"""

import torch
import csv, os

from math import log10, sqrt
from scipy.stats import norm

from operator import itemgetter
from typing import Callable, Type

import sys
import logging

sys.path.append("../..")
from gmm_utils import GMM, gmm_score_product, solve_max_ent_gmm, ScoreOffsetNetwork
from utils import kappa, m, sigma, BackwardDrift, compute_KL_est
from neural_sde import NeuralSDE, SamplePath

from cgm.cgm import calibrate_relaxed
from cgm.utils import clone_network, compute_violation_loss

def null_logger(*args): pass

def instantiate_base_model(dim: int, bprop: float, hstar: float, diff: float, std: float, hidden_dim: int, time_embed_dim: int, nsteps: int, device: Type[torch.device]) -> NeuralSDE:
    # Instantiate base GMM
    means = torch.tensor([[diff / 2], [-(diff / 2)]], device=device)
    cov = torch.tensor([[(std)**2]], device=device)
    mixture_props = torch.tensor([bprop, 1-bprop], device=device)
    gmm = GMM(means, cov, mixture_props)

    # Instantiate the base generative model
    ## Backward drift
    score_fn = lambda x, t: gmm_score_product(x, t, gmm, m_fn=m, sigma_fn=sigma) # no training is necessary since the score has a closed-form
    score_network = ScoreOffsetNetwork(score_fn, input_dim=dim, hidden_dim=hidden_dim, time_emb_dim=time_embed_dim, sigma=sigma, device=device)    
    forward_drift = lambda x, t: (-1 / 2) * kappa(t).unsqueeze(-1) * x
    forward_diffusion = lambda t: torch.sqrt(kappa(t))
    base_drift = BackwardDrift(score_network, forward_drift, forward_diffusion)
   
    ## Neural-SDE
    t_grid = torch.linspace(0, 1, nsteps + 1, device=device)
    base_model = NeuralSDE(
        sde_dim=dim,
        drift=base_drift,
        diffusion=(lambda t: torch.sqrt(kappa(1.0 - t))),
        t_grid=t_grid,
    )
    return base_model

def hh(x: SamplePath) -> torch.Tensor:
    x = x.xs[:, -1] # (batch_size, 1)
    return (x >= 0).to(torch.float32).reshape(-1, 1)

def compute_lambd(dim: int, pb: float, h_star: float, lambds: list[float], base_model: NeuralSDE, hh: Callable[[SamplePath], torch.Tensor], n_epochs: int, n_batches: int, batch_size: int, batch_size_eval: int, lr_init: float, lr_min: float, device: Type[torch.device], kappa: float = 0.1, logger = None, samp_chunks: int = 1, batch_chunks: int = 1) -> float:
    """
    Function for performing a grid search over values of lambda

    Returns the largest value of lambda for which the constraint violation is reduced by a factor of kappa; if no such value exists, returns the largest value of lambda
    """

    triples = []
    for lambd in lambds:
        if logger is not None:
            logger.info(f"Performing calibration with lambda={lambd}")
        
        calibrated_model = calibrate_relaxed(
            clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
            hh,
            torch.tensor([h_star], device=device).repeat(dim),
            lambd,
            n_epochs,
            batch_size,
            optimizer_params={"lr": lr_init},
            scheduler_params={"T_max": n_epochs, "eta_min": lr_min},
            logger=null_logger,
            samp_chunks = samp_chunks, 
            batch_chunks = batch_chunks,
            disable_pbar=True,
        ) 
        with torch.no_grad():
            xs = calibrated_model.sample(batch_size_eval)
            hx = hh(xs) # (batch_size_eval, dim)
            viol_loss = compute_violation_loss(hx, torch.tensor([h_star], device=device).repeat(dim), torch.ones(batch_size_eval, device=device)) # violation loss
            kl_loss = compute_KL_est(base_model, xs).mean()

        triples.append((lambd, kl_loss.item(), sqrt(torch.clip(viol_loss, min=0.).item())))
    
    viol_base = sqrt(dim) * abs(pb - h_star)

    # Candidates that satisfy the violation criterion
    admissible = [tpl for tpl in triples if tpl[2] < kappa * viol_base] # NOTE: we are comparing the sqrt of the violation loss
    
    if admissible:
        best_lambda, *_ = min(admissible, key=itemgetter(1))
    else:
        best_lambda, *_ = min(triples, key=itemgetter(1))
    
    return best_lambda # return optimal value of lambda

def main():

    # Configure logging level, format, etc.
    logger = logging.getLogger("pi_logger")
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Performing fine-tuning on device {device}")

    # Set output directory
    output_dir = "../../simulation_results/"
    os.makedirs(output_dir, exist_ok=True)

    # Write columns to output csv
    csv_path = os.path.join(output_dir, "rare_event_relax(1).csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p(X > 0)", "b_prop", "h_star", "min_kl", "lambda", "kl_est", "std_kl_est", "viol_est", "std_viol_est", "h_star_est", "std_h_star_est"])

    # Hyperparameters to fix
    STD = 0.25
    DIFF = 2.
    HSTAR = 0.8
    NSTEPS = 100
    N_REPS_CALIBRATE = 10 # number of calibration replicates
    N_EPOCHS = 2000
    N_BATCHES = 1
    BATCH_SIZE = 100
    BATCH_SIZE_EVAL = 10000
    HIDDEN_DIM = 256
    TIME_EMBED_DIM = 32
    LR_INIT = 1e-3
    LR_MIN = 1e-6
    KAPPA = 0.1

    lambds = torch.logspace(0, -3, 9 + 1).tolist() # list of lambda values to search over
    pb_start, pb_end = 2 * (1 - norm.cdf(DIFF / (2 * STD))), HSTAR * norm.cdf(DIFF / (2 * STD)) + (1 - HSTAR) * (1 - norm.cdf(DIFF / (2 * STD))) # smallest possible P(X > 0), largest possible P(X > 0) 
    props = torch.logspace(log10(pb_start), log10(pb_end), 20 + 1).tolist() # P(X > 0) under the base model; NOTE: prop is *not* equal to BPROP, except in the case BPROP = 0.5
    for prop in props:
        
        # Compute GMM mixture proportion from P(X > 0)
        BPROP = ((prop - 1 + norm.cdf(DIFF / (2 * STD))) / (2*norm.cdf(DIFF / (2 * STD)) - 1)).item()

        # Compute KL distance of max entropy solution to base model 
        alpha_star, min_kl = solve_max_ent_gmm(bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD)
        logger.info(f"Calibrating pre-trained model with prop={prop}: true alpha {alpha_star}, min KL {min_kl}")

        # Instantiate the base model
        base_model = instantiate_base_model(dim=1, bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD, hidden_dim=HIDDEN_DIM, time_embed_dim=TIME_EMBED_DIM, nsteps=NSTEPS, device=device)

        # Perform a grid search to choose the optimal lambda
        logger.info(f"Performing a grid search over {len(lambds)} values of lambda: min {min(lambds)}, max {max(lambds)}")
        lambd = compute_lambd(dim=1, pb=prop, h_star=HSTAR, lambds=lambds, base_model=base_model, hh=hh, n_epochs=N_EPOCHS, n_batches=N_BATCHES, batch_size=BATCH_SIZE, batch_size_eval=BATCH_SIZE_EVAL, lr_init=LR_INIT, lr_min=LR_MIN, device=device, kappa=KAPPA, logger=logger)
        logger.info(f"Calibrating pre-trained model with lambda={lambd}")

        kl_list, viol_list, h_est_list = [], [], []
        for i in range(N_REPS_CALIBRATE):
            calibrated_model = calibrate_relaxed(
                clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
                hh,
                torch.tensor([HSTAR], device=device),
                lambd,
                N_EPOCHS,
                BATCH_SIZE,
                optimizer_params={"lr": LR_INIT},
                scheduler_params={"T_max": N_EPOCHS, "eta_min": LR_MIN},
                logger=null_logger,
                disable_pbar=True
            ) 
            # Compute metrics using larger batch size
            with torch.no_grad():
                xs = calibrated_model.sample(BATCH_SIZE_EVAL)
                hx = hh(xs)
                viol_loss = compute_violation_loss(hx, torch.tensor([HSTAR], device=device), torch.ones(BATCH_SIZE_EVAL,  device=device))
                kl_loss = compute_KL_est(base_model, xs).mean()
                h_est = hx.mean()

            kl_list.append(kl_loss.item()); viol_list.append(viol_loss.item()); h_est_list.append(h_est.item())
            print(f"Finished iteration {i+1}/{N_REPS_CALIBRATE} of calibration")

        kls, viols, hs = torch.tensor(kl_list), torch.tensor(viol_list), torch.tensor(h_est_list)
        print(f"h estimate: {torch.mean(hs)} +/- {torch.std(hs)}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                prop,
                BPROP,
                HSTAR,
                min_kl,
                lambd, 
                torch.mean(kls).item(), 
                torch.std(kls).item(),
                torch.mean(viols).item(), 
                torch.std(viols).item(),
                torch.mean(hs).item(), 
                torch.std(hs).item()
                ])


if __name__ == "__main__":
    main()