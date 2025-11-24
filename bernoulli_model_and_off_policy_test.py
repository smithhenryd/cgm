import matplotlib.pyplot as plt
import numpy as np
import torch
from cgm.cgm import calibrate_relaxed, calibrate_relaxed_offpolicy

# Create a toy test case, a Bernoulli model with a single parameter that can be used a unit test for CGM

import torch.nn as nn

class BernoulliModel(nn.Module):
    def __init__(self, p):
        """
        Initialize the Bernoulli model with a probability parameter p.
        :param p: Initial probability of success (0 <= p <= 1)
        """
        super(BernoulliModel, self).__init__()
        self.logit_p = torch.nn.Parameter(torch.logit(
            torch.tensor(p, dtype=torch.float32)))
        self.device = torch.device("cpu")
    
    def model_p(self):
        return torch.sigmoid(self.logit_p)

    def sample(self, N):
        """
        Draw N independent samples from the Bernoulli model.
        :param N: Number of samples to draw
        :return: Tensor of samples with shape (N, 1)
        """
        samples = torch.bernoulli(self.model_p().expand(N))
        return samples.unsqueeze(-1)  # Add an extra dimension to match expected shape

    def log_p(self, x, **kwargs):
        """
        Compute the log probability of the samples x under the Bernoulli model.
        :param x: Tensor of samples (0s and 1s) with shape (N, 1)
        :return: Tensor of log probabilities with shape (N,)
        """
        x = torch.as_tensor(x, dtype=torch.float32).squeeze(-1)  # Remove extra dimension
        if not torch.all((x == 0) | (x == 1)):
            raise ValueError("Samples must only contain 0s and 1s.")
        p = self.model_p()

        return x * torch.log(p) + (1 - x) * torch.log(1 - p)

# Create a logger that saves the value of theta and phi at each iteration, 
class SimpleLogger:
    def __init__(self):
        self.theta = []
        self.phi = []
        self.h_bar = []  # NEW: track h_bar
        self.log_freq = 10  # Log every 10 epochs

    # create call method to log the values, it take a dictionary and then a number of arguments that can be ignored.
    def __call__(self, state, *args, **kwargs):

        # Estimate h_bar from the model
        if state['epoch'] % self.log_freq == 0:  # Log every 10 epochs
            if 'theta' in state: self.theta.append(state['theta'])
            if 'phi' in state: self.phi.append(state['phi'])
            with torch.no_grad():
                model = args[0]  # first arg is the model
                samples = model.sample(10000)
                h_bar_estimate = samples.mean().item()
                self.h_bar.append(h_bar_estimate)

            # Print all state values with precision of 2 decimals with nice formatting of 1 tab between key and value, all values on a single line
            print("\t".join([f"{key}: {value:.4f}" for key, value in state.items()]))

def test_calibrate_relaxed_off_policy(h_star=0.9, on_policy=False):
    """
    Test the off_policy calibrate_relaxed function on the Bernoulli model.
    """
    # Define the target probability
    h_star = torch.tensor([h_star], dtype=torch.float32)  # Target probability

    # Initialize the Bernoulli model with an initial probability
    model = BernoulliModel(p=0.5)

    # Calibrate the model using calibrate_relaxed
    logger = SimpleLogger()
    kwargs = {
        'model': model,
        'h': lambda x: x, # Indicator for x=1
        'hstar': h_star,
        'lambd': 0.,  # Regularization parameter
        'epochs': 500,  # Number of iterations
        'batch_size': 8,  # Number of samples per batch
        'optimizer_params': {"lr": 1e-1},  # Optimizer parameters
        'disable_pbar': True,
        'logger': logger, 
    }
    if on_policy:
        calibrated_model = calibrate_relaxed(**kwargs)
    else:
        calibrated_model = calibrate_relaxed_offpolicy(
            **kwargs,
        )

    # Check if the calibrated probability is close to the target
    calibrated_p = calibrated_model.model_p().item()
    if abs(calibrated_p - h_star.item()) < 0.05:
        print(f"PASS. Calibrated p={calibrated_p} is close to target h*={h_star.item()}.")
    else:
        print(f"FAIL. Calibrated p={calibrated_p} is not close to target h*={h_star.item()}.")
    return logger

# Run the test
if __name__ == "__main__":
    h_star = 0.95
    logger  = test_calibrate_relaxed_off_policy(h_star=h_star)
    
    # Plot h_bar and phi over iterations for a single run
    h_bar_vals = np.array(logger.h_bar)
    phi_vals = np.array(logger.phi)
    plt.plot(h_bar_vals, label='h_bar')
    if len(phi_vals) > 0:
        plt.plot(phi_vals, label='phi (logit q)')
    plt.axhline(y=h_star, color='r', linestyle='--', label='target h_star')
    plt.xlabel(f'Iteration (×{logger.log_freq})')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Calibration using Off-Policy CGM')
    plt.savefig(f'Bernoulli_calibration_offPolicy_hStar={h_star}.png', dpi=300)
    plt.clf()

    # Repeat several times both on_policy=True and on_policy=False to compare error bars
    off_policy_errors_list = []
    n_reps = 25
    for i in range(n_reps):
        logger_off = test_calibrate_relaxed_off_policy(h_star=h_star, on_policy=False)
        h_bar_off = np.array(logger_off.h_bar)

        errors = np.abs(h_bar_off - h_star)
        off_policy_errors_list.append(errors)
    off_policy_errors = np.array(off_policy_errors_list)
    off_policy_mean_errors = off_policy_errors.mean(axis=0)
    off_policy_sem_errors = off_policy_errors.std(axis=0) / np.sqrt(off_policy_errors.shape[0])

    # Repeat with on_policy=True
    on_policy_errors_list = []
    for i in range(n_reps):
        logger_on = test_calibrate_relaxed_off_policy(h_star=h_star, on_policy=True)
        h_bar_on = np.array(logger_on.h_bar)
        errors = np.abs(h_bar_on - h_star)
        on_policy_errors_list.append(errors)
    on_policy_errors = np.array(on_policy_errors_list)
    on_policy_mean_errors = on_policy_errors.mean(axis=0)
    on_policy_sem_errors = on_policy_errors.std(axis=0) / np.sqrt(on_policy_errors.shape[0])

    # Plot ||h_bar - h_star|| ± 2 SEM
    iterations = np.arange(len(off_policy_mean_errors))
    plt.fill_between(
        iterations,
        off_policy_mean_errors - 2 * off_policy_sem_errors,
        off_policy_mean_errors + 2 * off_policy_sem_errors,
        alpha=0.2,
        label='±2 SEM Off-Policy',
    )
    plt.plot(iterations, off_policy_mean_errors, label='Mean ||h_bar - h*|| Off-Policy')

    plt.fill_between(
        iterations,
        on_policy_mean_errors - 2 * on_policy_sem_errors,
        on_policy_mean_errors + 2 * on_policy_sem_errors,
        alpha=0.2,
        label='±2 SEM On-Policy',
    )
    plt.plot(iterations, on_policy_mean_errors, label='Mean ||h_bar - h*|| On-Policy')

    plt.xlabel(f'Iteration (×{logger.log_freq})')
    plt.ylabel('|h_bar - h*|')
    plt.yscale('log')
    plt.legend()
    plt.title('On-Policy vs Off-Policy Calibration Error over Iterations')
    # Save the plot as a PNG file
    plt.savefig(f'Bernoulli_error_comparison_hStar={h_star}.png', dpi=300)