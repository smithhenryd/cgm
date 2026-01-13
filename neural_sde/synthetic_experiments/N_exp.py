"""
Script for running CGM-reward for various values of N 
Reproduces Figure 1A in the paper
"""

import torch
import csv, os

from math import log10

import sys
sys.path.append("..")
import logging

from gmm_utils import GMM, gmm_score, solve_max_ent_gmm, ScoreOffsetNetwork
from utils import kappa, m, sigma, BackwardDrift, compute_KL_est
from neural_sde import NeuralSDE, SamplePath

from cgm.cgm import calibrate_reward
from cgm.utils import clone_network, compute_violation_loss

def null_logger(*args): pass

def hh(x: SamplePath) -> torch.Tensor:
    x = x.xs[:, -1] # (batch_size, 1)
    return (x >= 0).to(torch.float32).reshape(-1, 1)

def main():

    # Configure logging level, format, etc.
    logger = logging.getLogger("N_logger")
    logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Performing fine-tuning on device {device}")

    # Set output directory
    output_dir = "../simulation_results/"
    os.makedirs(output_dir, exist_ok=True)

    # Write columns to output csv
    csv_path = os.path.join(output_dir, "N_exp.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "kl_est", "std_kl_est", "viol_est", "std_viol_est", "h_star_est", "std_h_star_est"])

    # Hyperparameters to fix
    STD = 0.25
    DIFF = 2.
    BPROP = 0.5
    HSTAR = 0.8
    NSTEPS = 100
    N_REPS_CALIBRATE = 10 # number of calibration replications
    N_EPOCHS = 2000
    N_BATCHES = 1
    BATCH_SIZE = 100
    BATCH_SIZE_EVAL = 10000
    HIDDEN_DIM = 256
    TIME_EMBED_DIM = 32
    LR_INIT = 1e-3
    LR_MIN = 1e-6

    # Compute analytical solution for alpha* and min KL
    alpha_star, min_kl = solve_max_ent_gmm(bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD)
    logger.info(f"Solved max entropy problem: true alpha {alpha_star}, min KL {min_kl}")

    # Instantiate base GMM
    means = torch.tensor([[DIFF / 2], [-(DIFF / 2)]], device=device)
    cov = torch.tensor([[(STD)**2]], device=device)
    mixture_props = torch.tensor([BPROP, 1-BPROP], device=device)
    gmm = GMM(means, cov, mixture_props)

    # Instantiate the base generative model
    ## Backward drift
    score_fn = lambda x, t: gmm_score(x, t, gmm, m_fn=m, sigma_fn=sigma) # no training is necessary since the score has a closed-form
    score_network = ScoreOffsetNetwork(score_fn, input_dim=1, hidden_dim=HIDDEN_DIM, time_emb_dim=TIME_EMBED_DIM, sigma=sigma, device=device)    
    forward_drift = lambda x, t: (-1 / 2) * kappa(t).unsqueeze(-1) * x
    forward_diffusion = lambda t: torch.sqrt(kappa(t))
    base_drift = BackwardDrift(score_network, forward_drift, forward_diffusion)
   
    ## Neural-SDE
    t_grid = torch.linspace(0, 1, NSTEPS + 1, device=device)
    base_model = NeuralSDE(
        sde_dim=1,
        drift=base_drift,
        diffusion=(lambda t: torch.sqrt(kappa(1.0 - t))),
        t_grid=t_grid,
    )

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            None, 
            min_kl,
            None, 
            0.,
            None,
            HSTAR, 
            None])

    Ns = torch.logspace(log10(5), 5, 30 + 1) # number of samples for estimating alpha*
    for N in Ns:
        N = int(N)
        logger.info(f"Calibrating pre-trained model with N={N}")
        kl_list, viol_list, h_est_list = [], [], []

        try:
            for i in range(N_REPS_CALIBRATE):
                calibrated_model = calibrate_reward(
                    clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
                    hh,
                    torch.tensor([HSTAR], device=device),
                    N,
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
                    N, 
                    torch.mean(kls).item(), 
                    torch.std(kls).item(),
                    torch.mean(viols).item(), 
                    torch.std(viols).item(),
                    torch.mean(hs).item(), 
                    torch.std(hs).item()
                    ])
        except ValueError: # skip N if the max entropy problem is infeasible
            logger.warning("Max entropy problem is infeasible, skipping N")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                        N, 
                        None, 
                        None,
                        None, 
                        None,
                        None, 
                        None
                        ])
            continue


if __name__ == "__main__":
    main()