"""
Analysis of expected gradient and implied loss for Squared error, Absolute error,
and Stationary REINFORCE estimators on the Bernoulli model.

For a Bernoulli(p) model with h(x)=x, parameter p = E[h(x)].
We compute E_p[U_hat] exactly (finite sum over binomial outcomes) and
integrate to get the implied loss V(p) in p-space:

    dV/dp = E_p[U_hat] / (p*(1-p))    [chain rule: d/dp = d/d(logit) * 1/(p(1-p))]

so  V(p) = integral_0^p  E_q[U_hat] / (q*(1-q)) dq.

NOTE: The Stationary REINFORCE formula requires M large enough that multiple
discrete outcomes k/M lie above h*.  For h*=0.9 this requires M >= 10;
M=32 is used here to ensure validity across all h* values shown.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.integrate import cumulative_trapezoid


# ── helpers ───────────────────────────────────────────────────────────────────

def binom_pmf(M, p):
    """PMF array P[X=k] for X~Binomial(M,p), shape (M+1,)."""
    k = np.arange(M + 1, dtype=float)
    log_pmf = (gammaln(M + 1) - gammaln(k + 1) - gammaln(M - k + 1)
               + k * np.log(np.clip(p, 1e-300, 1))
               + (M - k) * np.log(np.clip(1 - p, 1e-300, 1)))
    return np.exp(log_pmf)


def compute_weights(hstar, M):
    """Compute Stationary REINFORCE weights (w0, w1) from h* and batch size M.

    Solves for weights satisfying:
      (i)  w(1) - w(0) = 1   (normalization; matches absolute-error scaling far from h*)
      (ii) E_{h*}[U_hat] = 0  (stationarity; zero expected update at p = h*)

    The solution is:
      w(0) = -(1-h*) * w0_factor / D
      w(1) =  h*     * w1_factor / D
      D    =  2*alpha*(h* - beta)

    where  w0_factor = h* - 2*alpha*beta       (> 0 always, since beta < h*)
           w1_factor = 2*alpha*(1-beta) - (1-h*)  (> 0 when M is not too large relative to 1/h*)

    """
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, hstar)
    mask = k / M < hstar
    alpha = pmf[mask].sum()
    assert alpha > 1e-15, "unstable if alpha < 1e-15"
    beta = ((k[mask] / M) * pmf[mask]).sum() / alpha
    w0_factor = hstar - 2 * alpha * beta                 # h* - 2αβ; > 0 always
    w1_factor = 2 * alpha * (1 - beta) - (1 - hstar)    # 2α(1-β) - (1-h*); > 0 for M not too large
    D = 2 * alpha * (hstar - beta)
    w0 = -(1 - hstar) * w0_factor / D
    w1 = hstar * w1_factor / D
    return w0, w1

def expected_gradient_L2(p, hstar, M):
    return 2*(p-hstar)

def expected_gradient_L1(p, hstar, M):
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, p)
    y = k / M
    sign = np.sign(y - hstar)
    G = sign * (y * (1-hstar) *(1/p) - (1 - y) * (-hstar) * (1/(1-p)))
    return (pmf * G).sum()


def expected_gradient_SR(p, hstar, M, w0, w1):
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, p)
    y = k / M
    sign = np.where(y >= hstar, 1.0, -1.0)
    G = sign / M * (k * w1 * (1/p) - (M - k) * w0  * (1/(1-p)))
    return (pmf * G).sum()


# ── main analysis ─────────────────────────────────────────────────────────────
M_large = 64
M_small = 8
hstar_list = [0.01, 0.1, 0.3]
p_grid = np.linspace(0.005, 0.995, 600)

# Colors per estimator; line style encodes batch size
estimator_colors = {
    'Squared error':        'red',
    'Absolute error':       '0.5',
    'Stationary REINFORCE': 'b',
}
# (M, linestyle) pairs for the two batch-size-dependent estimators
batch_conditions = [(M_large, ':'), (M_small, '-')]

fig, axes = plt.subplots(2, len(hstar_list), figsize=(19, 6), dpi=150)

for j, hstar in enumerate(hstar_list):
    # ── Squared error: batch-size independent, plot once ──────────────────
    color = estimator_colors['Squared error']
    EG = np.array([expected_gradient_L2(p, hstar, M_large) for p in p_grid])
    axes[0, j].plot(p_grid, EG, color=color, linestyle='-', linewidth=1.0)
    V = cumulative_trapezoid(EG, p_grid, initial=0)
    V -= V[np.argmin(np.abs(p_grid - hstar))]
    axes[1, j].plot(p_grid, V, color=color, linestyle='-', linewidth=1.0)

    # ── Absolute error and Stationary REINFORCE: two batch sizes ──────────
    for M, ls in batch_conditions:
        w0, w1 = compute_weights(hstar, M)
        print(f"h*={hstar}, M={M}: w(0) = {w0:.4f}, w(1) = {w1:.4f}")

        for name, fn in [
            ('Absolute error',       lambda p, h, w0, w1, M=M: expected_gradient_L1(p, h, M)),
            ('Stationary REINFORCE', lambda p, h, w0, w1, M=M: expected_gradient_SR(p, h, M, w0, w1)),
        ]:
            color = estimator_colors[name]
            EG = np.array([fn(p, hstar, w0, w1) for p in p_grid])
            axes[0, j].plot(p_grid, EG, color=color, linestyle=ls, linewidth=1.0)
            V = cumulative_trapezoid(EG, p_grid, initial=0)
            V -= V[np.argmin(np.abs(p_grid - hstar))]
            axes[1, j].plot(p_grid, V, color=color, linestyle=ls, linewidth=1.0)

    for row in range(2):
        axes[row, j].axhline(0, color='k', linewidth=0.5)
        axes[row, j].axvline(hstar, color='k', linewidth=0.5)
        axes[row, j].set_xlim([0, 1])

    axes[0, j].set_title(f'h* = {hstar}', fontsize=22)
    axes[0, j].set_xlabel('p', fontsize=20)
    axes[1, j].set_xlabel('p', fontsize=20)
    for row in range(2):
        axes[row, j].tick_params(labelsize=17)

axes[0, 0].set_ylabel(r'$\mathbb{E}_p[-\hat{U}]$', fontsize=22)
axes[1, 0].set_ylabel('V(p) \n (implied loss)', fontsize=22)

# ── Legend: estimator colors + batch-size line styles ─────────────────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='red',  linestyle='-',  linewidth=1.0, label='Squared error'),
    Line2D([0], [0], color='0.5',  linestyle='-',  linewidth=1.0, label='Absolute error'),
    Line2D([0], [0], color='b',    linestyle='-',  linewidth=1.0, label='Stationary REINFORCE'),
    Line2D([0], [0], color='k',    linestyle='-',  linewidth=1.0, label=f'M={M_small}'),
    Line2D([0], [0], color='k',    linestyle=':',  linewidth=1.0, label=f'M={M_large}'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=len(legend_handles),
           fontsize=19, frameon=False, bbox_to_anchor=(0.5, 1.00))

plt.suptitle('Expected gradient and implied loss — Bernoulli model',
             fontsize=22, y=1.03)
plt.tight_layout()
plt.savefig('sr_analysis.pdf', bbox_inches='tight')
plt.show()
print("Saved sr_analysis.pdf")
