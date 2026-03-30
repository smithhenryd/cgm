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

    When w1_factor <= 0 the finite-M formula is invalid because it would give w(1) < 0,
    making the update diverge. In this regime D, w0_factor, and w1_factor all vanish
    at the same O(sigma) rate as M -> inf (by CLT), so by L'Hopital the weights
    converge to w(0) -> -(1-h*) and w(1) -> h*. These are returned as the fallback.
    """
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, hstar)
    mask = k / M < hstar
    alpha = pmf[mask].sum()
    if alpha < 1e-15:
        return -(1 - hstar), hstar
    beta = ((k[mask] / M) * pmf[mask]).sum() / alpha
    w0_factor = hstar - 2 * alpha * beta                 # h* - 2αβ; > 0 always
    w1_factor = 2 * alpha * (1 - beta) - (1 - hstar)    # 2α(1-β) - (1-h*); > 0 for M not too large
    if w0_factor < 1e-15 or w1_factor < 1e-15:
        # L'Hopital (M -> inf) limit: all three quantities vanish at O(sigma) rate
        return -(1 - hstar), hstar
    D = 2 * alpha * (hstar - beta)
    w0 = -(1 - hstar) * w0_factor / D
    w1 = hstar * w1_factor / D
    return w0, w1


# ── gradient formulas (per-batch, then averaged over batches) ─────────────────
#
# For a batch with k ones out of M, gradient w.r.t. logit = sum_m c_m*(h_m - p).
#
# L2 (bias-corrected, as in compute_violation_loss):
#   loss = (y-h*)^2 - (1/(M(M-1))) sum_m (h_m - y)^2
#   c_m  = 2(h_m-h*)/M * [(y-h*) - (h_m-y)/(M-1)]
#   => G = 2(y-h*)[y(1-h*)(1-p) + (1-y)h*p] + 2(h*+p-1)*y*(1-y)/(M-1)
#   E_p[G] = 2(p-h*)*p*(1-p)  [= gradient of (p-h*)^2, zero at p=h*]
#
# L1:  c_m = sign(y - h*) * (h_m - h*) / M
#      => G = sign(y-h*) * [y*(1-h*)*(1-p) + (1-y)*h*p]
#
# SR:  c_m = (-1)^I[y<h*] * w(h_m) / M,  w(0)=w0, w(1)=w1  (see compute_weights)
#      => G = (-1)^I[y<h*] / M * [k*w1*(1-p) - (M-k)*w0*p]

def expected_gradient_L2(p, hstar, M):
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, p)
    y = k / M
    G_main = 2 * (y - hstar) * (y * (1 - hstar) * (1 - p) + (1 - y) * hstar * p)
    G_bias  = 2 * (hstar + p - 1) * y * (1 - y) / (M - 1)
    return (pmf * (G_main + G_bias)).sum()


def expected_gradient_L1(p, hstar, M):
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, p)
    y = k / M
    sign = np.sign(y - hstar)
    G = sign * (y * (1 - hstar) * (1 - p) + (1 - y) * hstar * p)
    return (pmf * G).sum()


def expected_gradient_SR(p, hstar, M, w0, w1):
    k = np.arange(M + 1, dtype=float)
    pmf = binom_pmf(M, p)
    y = k / M
    sign = np.where(y >= hstar, 1.0, -1.0)
    G = sign / M * (k * w1 * (1 - p) - (M - k) * w0 * p)
    return (pmf * G).sum()


# ── main analysis ─────────────────────────────────────────────────────────────

M = 16
hstar_list = [0.01, 0.1, 0.3, 0.5, 0.9]
p_grid = np.linspace(0.005, 0.995, 600)

estimators = [
    ('Squared error',        'k',   '--', lambda p, h, w0, w1: expected_gradient_L2(p, h, M)),
    ('Absolute error',       '0.5', '-',  lambda p, h, w0, w1: expected_gradient_L1(p, h, M)),
    ('Stationary REINFORCE', 'b',   '-',  lambda p, h, w0, w1: expected_gradient_SR(p, h, M, w0, w1)),
]

fig, axes = plt.subplots(2, len(hstar_list), figsize=(4+3*len(hstar_list), 6), dpi=150)

for j, hstar in enumerate(hstar_list):
    w0, w1 = compute_weights(hstar, M)
    print(f"h*={hstar}, M={M}: w(0) = {w0:.4f}, w(1) = {w1:.4f}")

    for name, color, ls, fn in estimators:
        EG = np.array([fn(p, hstar, w0, w1) for p in p_grid])

        # Row 1: E[G_hat] vs p
        axes[0, j].plot(p_grid, EG, label=name, color=color, linestyle=ls, linewidth=1.5)

        # Row 2: Implied loss V(p)
        # V'(p) = E_p[G_hat] / (p*(1-p))
        integrand = EG / (p_grid * (1 - p_grid))
        V = cumulative_trapezoid(integrand, p_grid, initial=0)
        idx_hstar = np.argmin(np.abs(p_grid - hstar))
        V -= V[idx_hstar]   # shift so V(h*) = 0
        axes[1, j].plot(p_grid, V, label=name, color=color, linestyle=ls, linewidth=1.5)

    for row in range(2):
        axes[row, j].axhline(0, color='gray', linewidth=0.5)
        axes[row, j].axvline(hstar, color='r', linewidth=1, linestyle=':', alpha=0.7)
        axes[row, j].set_xlim([0, 1])

    axes[0, j].set_title(f'h* = {hstar}', fontsize=10)
    axes[0, j].set_xlabel('p')
    axes[1, j].set_xlabel('p')

axes[0, 0].set_ylabel(r'$\mathbb{E}_p[\hat{U}]$', fontsize=11)
axes[1, 0].set_ylabel('V(p)  (implied loss)', fontsize=11)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(estimators),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.97))

plt.suptitle(f'Expected gradient and implied loss — Bernoulli model, M={M}',
             fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig('sr_analysis.pdf', bbox_inches='tight')
plt.show()
print("Saved implied_loss_analysis.pdf")
