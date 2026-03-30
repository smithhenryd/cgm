"""
Comparison of Squared Error vs Stationary REINFORCE on the Bernoulli model.

Grid: h* in {0.1, 0.01, 0.001} x batch_size in {8, 32, 128}, 4 replicates each.
Plots calibration error |E[h] - h*| vs gradient steps (log-log).
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from cgm import cgm


# ── Model and logger ──────────────────────────────────────────────────────────

class BernoulliModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.logit_p = nn.Parameter(torch.logit(torch.tensor(p, dtype=torch.float32)))
        self.device = torch.device("cpu")

    def model_p(self):
        return torch.sigmoid(self.logit_p)

    def sample(self, N):
        return torch.bernoulli(self.model_p().expand(N)).unsqueeze(-1)

    def log_p(self, x, **kwargs):
        x = torch.as_tensor(x, dtype=torch.float32).squeeze(-1)
        p = self.model_p()
        return x * torch.log(p) + (1 - x) * torch.log(1 - p)


class Logger:
    log_freq = 10

    def __init__(self):
        self.h_bar = []

    def __call__(self, state, *args, **kwargs):
        if state['epoch'] % self.log_freq == 0:
            with torch.no_grad():
                model = args[0]
                self.h_bar.append(model.sample(10000).mean().item())


# ── Calibration runners ───────────────────────────────────────────────────────

def run_squared_error(h_star, batch_size, epochs=5000, lr=3e-2):
    model = BernoulliModel(p=0.5)
    logger = Logger()
    cgm.calibrate_relaxed(
        model=model,
        h=lambda x: x,
        hstar=torch.tensor([h_star], dtype=torch.float32),
        lambd=0.,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_params={"lr": lr},
        disable_pbar=True,
        logger=logger,
        lr_scheduler_cls=None,
        L1_loss=False,
    )
    return np.array(logger.h_bar)


def run_stationary_reinforce(h_star, batch_size, epochs=5000, lr=3e-2):
    model = BernoulliModel(p=0.5)
    logger = Logger()
    cgm.calibrate_stationary_reinforce(
        model=model,
        h=lambda x: x,
        hstar=torch.tensor([h_star], dtype=torch.float32),
        lambd=0.,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_params={"lr": lr},
        disable_pbar=True,
        logger=logger,
        lr_scheduler_cls=None,
    )
    return np.array(logger.h_bar)


# ── Grid ──────────────────────────────────────────────────────────────────────

hstar_vals  = [0.1, 0.01, 0.001]
batch_sizes = [8, 32, 128]
n_reps      = 4
epochs      = 5000

methods = [
    ("Squared error",        "k", "--", run_squared_error),
    ("Stationary REINFORCE", "b", "-",  run_stationary_reinforce),
]

# ── Run simulations ───────────────────────────────────────────────────────────

results = {}   # (method_name, h_star, batch_size) -> (mean_errors, sem_errors)

total = len(hstar_vals) * len(batch_sizes) * len(methods) * n_reps
done = 0
for h_star in hstar_vals:
    for batch_size in batch_sizes:
        for method_name, _, _, runner in methods:
            h_bar_list = []
            for rep in range(n_reps):
                done += 1
                print(f"[{done}/{total}] h*={h_star}, M={batch_size}, {method_name}, rep {rep+1}")
                h_bar = runner(h_star, batch_size, epochs=epochs)
                h_bar_list.append(np.abs(h_bar - h_star))
            arr = np.array(h_bar_list)
            results[(method_name, h_star, batch_size)] = (
                arr.mean(axis=0),
                arr.std(axis=0) / np.sqrt(n_reps),
            )

# ── Plot ──────────────────────────────────────────────────────────────────────

n_rows = len(hstar_vals)
n_cols = len(batch_sizes)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=150)

log_freq = Logger.log_freq
for i, h_star in enumerate(hstar_vals):
    for j, batch_size in enumerate(batch_sizes):
        ax = axes[i, j]
        for method_name, color, ls, _ in methods:
            mean_err, sem_err = results[(method_name, h_star, batch_size)]
            iterations = np.arange(len(mean_err)) * log_freq
            ax.fill_between(
                iterations,
                np.maximum(mean_err - 2 * sem_err, 1e-9),
                mean_err + 2 * sem_err,
                alpha=0.15, color=color,
            )
            ax.plot(iterations, mean_err, label=method_name,
                    color=color, linestyle=ls, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([log_freq, epochs])
        ax.set_ylim(bottom=1e-4)
        ax.axhline(0, color='gray', linewidth=0.3)
        ax.grid(True, which="both", ls="--", linewidth=0.4, alpha=0.5)
        ax.set_title(f'$h^*={h_star}$, $M={batch_size}$', fontsize=9)
        ax.set_xlabel('Gradient steps', fontsize=8)
        ax.set_ylabel(r'$|\mathbb{E}_{p_\theta}[h(x)] - h^*|$', fontsize=8)
        if i == 0 and j == 0:
            ax.legend(frameon=False, fontsize=8)

plt.suptitle(
    'Squared Error vs Stationary REINFORCE — Bernoulli calibration\n'
    f'({n_reps} replicates, mean ± 2 SE)',
    fontsize=11,
)
plt.tight_layout()
plt.savefig('bernoulli_comparison.pdf', bbox_inches='tight')
plt.savefig('bernoulli_comparison.png', dpi=150, bbox_inches='tight')
print("Saved bernoulli_comparison.pdf / .png")
plt.show()
