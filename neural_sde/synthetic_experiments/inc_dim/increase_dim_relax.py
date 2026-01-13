"""
Script for calibrating a d-dimensional GMM to a d-dimensional constraint with CGM-relax
Reproduces Figure 1B in the paper
"""

import torch
import csv, os

from math import log
from scipy.stats import norm

import sys
import logging

sys.path.append("../..")
from gmm_utils import solve_max_ent_gmm
from utils import compute_KL_est
from neural_sde import SamplePath

from cgm.cgm import calibrate_relaxed
from cgm.utils import clone_network, compute_violation_loss

sys.path.append("../rare_event")
from rare_event_relax import compute_lambd, instantiate_base_model

def null_logger(*args): pass

def hh(x: SamplePath) -> torch.Tensor: # NOTE: h specifies a d-dimensional constraint
    x = x.xs[:, -1] # (batch_size, d)
    return (x >= 0).to(torch.float32)

def compute_hstar(pb: float, target: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """
    A helper function to solve for hstar that achieves target KL

    pb is P(X > 0) under the base model
    """
    def bern_kl(p, q):
        "KL between Bern(p) and Bern(q)"
        return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))

    if not (0 < pb < 1):
        raise ValueError("pb must be in (0,1)")
    
    kl_max = -log(1 - pb) if pb > 0.5 else -log(pb)
    if not (0 < target <= kl_max + 1e-15):
        raise ValueError(f"target_kl must be in (0, {kl_max}] for this b_prop")

    lo, hi = pb, 1 - 1e-15          
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        kl_mid = bern_kl(mid, pb)
        if abs(kl_mid - target) < tol:
            return mid
        if kl_mid < target:
            lo = mid
        else:
            hi = mid
    raise RuntimeError("Root finder did not converge")

def main():

    # Configure logging level, format, etc.
    logger = logging.getLogger("dim_logger")
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
    csv_path = os.path.join(output_dir, "inc_dim_relax.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dim", "b_prop", "h_star", "min_kl", "lambda", "kl_est", "std_kl_est", "viol_est", "std_viol_est"])

    # Hyperparameters to fix
    STD = 0.25
    DIFF = 2.
    BPROP = 0.5
    KL_TARGET = 0.2 # target KL divergence (if KL is fixed)
    NSTEPS = 100
    N_REPS_CALIBRATE = 10 # number of calibration replications
    N_EPOCHS = 2000
    N_BATCHES = 1
    BATCH_SIZE = 10000 # NOTE: batch size here is larger than in rate event experiments
    BATCH_SIZE_EVAL = 10000
    HIDDEN_DIM = 256
    TIME_EMBED_DIM = 32
    LR_INIT = 1e-3
    LR_MIN = 1e-6
    KAPPA = 0.1
    FIX_KL = False # if true, KL of max entropy solution to the base model is kept constant as the problem dimension increases

    # Compute KL distance of max entropy solution to base model, if KL grows with problem dimension
    pb = BPROP * norm.cdf(DIFF / (2 * STD)) + (1 - BPROP) * (1 - norm.cdf(DIFF / (2 * STD))) # P(X >= 0) under p_{base}; NOTE: prop is *not* equal to BPROP, except in the case BPROP = 0.5
    if not FIX_KL:
        HSTAR = compute_hstar(pb=pb, target=KL_TARGET)
        alpha_star, min_kl = solve_max_ent_gmm(bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD)

    lambds = torch.logspace(0, -3, 9 + 1).tolist() # list of lambda values to search over
    dims = torch.linspace(0, 1000, 20 + 1).tolist() # dimension of GMM + constraint
    dims[0] = 1
    for dim in dims:
        dim = int(dim)
        
        # Compute KL distance of max entropy solution to base model, if KL is held constant with problem dimension
        if FIX_KL: 
            HSTAR = compute_hstar(pb=pb, target=KL_TARGET / dim)
            alpha_star, min_kl = solve_max_ent_gmm(bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD)
        logger.info(f"Calibrating pre-trained model with dim={dim}: true alpha {alpha_star}, min KL {dim * min_kl}")

        # Instantiate the base model
        base_model = instantiate_base_model(dim=dim, bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD, hidden_dim=HIDDEN_DIM, time_embed_dim=TIME_EMBED_DIM, nsteps=NSTEPS, device=device)
        
        # Perform a grid search to choose the optimal lambda
        logger.info(f"Performing a grid search over {len(lambds)} values of lambda: min {min(lambds)}, max {max(lambds)}")
        if dim > 500:
            samp_chunks, batch_chunks = 1, 2
        else:
            samp_chunks, batch_chunks = 1, 1
        lambd = compute_lambd(dim=dim, pb=pb, h_star=HSTAR, lambds=lambds, base_model=base_model, hh=hh, n_epochs=N_EPOCHS, n_batches=N_BATCHES, batch_size=BATCH_SIZE, batch_size_eval=BATCH_SIZE_EVAL, lr_init=LR_INIT, lr_min=LR_MIN, device=device, kappa=KAPPA, logger=logger, samp_chunks=samp_chunks, batch_chunks=batch_chunks)
        logger.info(f"Calibrating pre-trained model with lambda={lambd}")

        kl_list, viol_list = [], []
        for i in range(N_REPS_CALIBRATE):
            calibrated_model = calibrate_relaxed(
                clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
                hh,
                torch.tensor([HSTAR], device=device).repeat(dim),
                lambd,
                N_EPOCHS,
                BATCH_SIZE,
                optimizer_params={"lr": LR_INIT},
                scheduler_params={"T_max": N_EPOCHS, "eta_min": LR_MIN},
                logger=null_logger,
                samp_chunks=samp_chunks,
                batch_chunks=batch_chunks,
                disable_pbar=True
            ) 
            # Compute metrics using larger batch size
            with torch.no_grad():
                xs = calibrated_model.sample(BATCH_SIZE_EVAL)
                hx = hh(xs) # (batch_size, dim)
                viol_loss = compute_violation_loss(hx, torch.tensor([HSTAR], device=device).repeat(dim), torch.ones(BATCH_SIZE_EVAL,  device=device)) # violation loss
                kl_loss = compute_KL_est(base_model, xs).mean()

            kl_list.append(kl_loss.item()); viol_list.append(viol_loss.item())
            print(f"Finished iteration {i+1}/{N_REPS_CALIBRATE} of calibration")

        kls, viols = torch.tensor(kl_list), torch.tensor(viol_list)
        print(f"viol estimate: {torch.mean(viols)} +/- {torch.std(viols)}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                dim,
                BPROP,
                HSTAR,
                dim * min_kl, # dim * min_kl since the max entropy distribution is independent across dimensions
                lambd, 
                torch.mean(kls).item(), 
                torch.std(kls).item(),
                torch.mean(viols).item(), 
                torch.std(viols).item(),
                ])


if __name__ == "__main__":
    main()