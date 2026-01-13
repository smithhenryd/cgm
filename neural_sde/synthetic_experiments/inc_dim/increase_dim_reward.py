"""
Script for calibrating a d-dimensional GMM to a d-dimensional constraint with CGM-reward
Reproduces Figure 1B in the paper
"""

import torch
import csv, os

from scipy.stats import norm

import sys
import logging

sys.path.append("../..")
from gmm_utils import solve_max_ent_gmm
from utils import compute_KL_est
from neural_sde import SamplePath

from cgm.cgm import calibrate_reward
from cgm.utils import clone_network, compute_violation_loss

from increase_dim_relax import compute_hstar
sys.path.append("../rare_event")
from rare_event_relax import instantiate_base_model

def null_logger(*args): pass

def hh(x: SamplePath) -> torch.Tensor: # NOTE: h specifies a d-dimensional constraint
    x = x.xs[:, -1] # (batch_size, d)
    return (x >= 0).to(torch.float32)

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
    csv_path = os.path.join(output_dir, "inc_dim_reward.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dim", "b_prop", "h_star", "min_kl", "kl_est", "std_kl_est", "viol_est", "std_viol_est"])

    # Hyperparameters to fix
    STD = 0.25
    DIFF = 2.
    BPROP = 0.5
    KL_TARGET = 0.2 # target KL divergence (if KL is fixed)
    NSTEPS = 100
    N_REPS = 100000
    N_REPS_CALIBRATE = 10 # number of calibration replications
    N_EPOCHS = 2000
    N_BATCHES = 1
    BATCH_SIZE = 10000 # NOTE: batch size here is larger than in rate event experiments
    BATCH_SIZE_EVAL = 10000
    HIDDEN_DIM = 256
    TIME_EMBED_DIM = 32
    LR_INIT = 1e-3
    LR_MIN = 1e-6
    FIX_KL = False # if true, KL of max entropy solution to the base model is kept constant as the problem dimension increases

    # Compute KL distance of max entropy solution to base model, if KL grows with problem dimension
    pb = BPROP * norm.cdf(DIFF / (2 * STD)) + (1 - BPROP) * (1 - norm.cdf(DIFF / (2 * STD))) # P(X >= 0) under p_{base}; NOTE: prop is *not* equal to BPROP, except in the case BPROP = 0.5
    if not FIX_KL:
        HSTAR = compute_hstar(pb=pb, target=KL_TARGET)
        alpha_star, min_kl = solve_max_ent_gmm(bprop=BPROP, hstar=HSTAR, diff=DIFF, std=STD)

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

        kl_list, viol_list = [], []
        if dim > 500:
            samp_chunks, batch_chunks = 1, 2
        else:
            samp_chunks, batch_chunks = 1, 1
        try:
            for i in range(N_REPS_CALIBRATE):
                calibrated_model = calibrate_reward(
                    clone_network(base_model, disable_gradients=False),  # clone the base model so we can keep our original copy of it
                    hh,
                    torch.tensor([HSTAR], device=device).repeat(dim),
                    N_REPS,
                    alpha_star,
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
                    torch.mean(kls).item(), 
                    torch.std(kls).item(),
                    torch.mean(viols).item(), 
                    torch.std(viols).item(),
                    ])
        except ValueError: # stop script if max ent problem is infeasible (since dim will only increase)
            logger.warning("Max entropy problem is infeasible, stopping script")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    dim, 
                    BPROP,
                    HSTAR,
                    dim * min_kl, 
                    None,
                    None,
                    None,
                    None
                    ])
            import sys; sys.exit()

    
if __name__ == "__main__":
    main()