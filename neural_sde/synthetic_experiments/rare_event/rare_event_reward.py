"""
Script for upweighting a rare mode in a GMM with CGM-reward
Reproduces Figure 1B in the paper
"""

import torch
import csv, os

from math import log10
from scipy.stats import norm

import sys
import logging

sys.path.append("../..")
from gmm_utils import solve_max_ent_gmm
from utils import compute_KL_est
from neural_sde import SamplePath

from cgm.cgm import calibrate_reward
from cgm.utils import clone_network, compute_violation_loss

from rare_event_relax import instantiate_base_model

def null_logger(*args): pass

def hh(x: SamplePath) -> torch.Tensor:
    x = x.xs[:, -1] # (batch_size, 1)
    return (x >= 0).to(torch.float32).reshape(-1, 1)

def main():

    # Configure logging level, format, etc.
    logger = logging.getLogger("pi_logger_reward")
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
    csv_path = os.path.join(output_dir, "rare_event_reward.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p(X > 0)", "b_prop", "h_star", "min_kl",  "kl_est", "std_kl_est", "viol_est", "std_viol_est", "h_star_est", "std_h_star_est"])

    # Hyperparameters to fix
    STD = 0.25
    DIFF = 2.
    HSTAR = 0.8
    NSTEPS = 100
    N_REPS = 100000 # number of reps to estimate alpha
    N_REPS_CALIBRATE = 10 # number of calibration replicates
    N_EPOCHS = 2000
    N_BATCHES = 1
    BATCH_SIZE = 100
    BATCH_SIZE_EVAL = 10000
    HIDDEN_DIM = 256
    TIME_EMBED_DIM = 32
    LR_INIT = 1e-3
    LR_MIN = 1e-6

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

        kl_list, viol_list, h_est_list = [], [], []
        try:
            for i in range(N_REPS_CALIBRATE):
                calibrated_model = calibrate_reward(
                    clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
                    hh,
                    torch.tensor([HSTAR], device=device),
                    N_REPS,
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
                    torch.mean(kls).item(), 
                    torch.std(kls).item(),
                    torch.mean(viols).item(), 
                    torch.std(viols).item(),
                    torch.mean(hs).item(), 
                    torch.std(hs).item()
                    ])
        except ValueError:
            logger.warning("Max entropy problem is infeasible, skipping pi")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    prop,
                    BPROP,
                    HSTAR,
                    min_kl,
                    None, 
                    None,
                    None, 
                    None,
                    None, 
                    None
                    ])


if __name__ == "__main__":
    main()