import torch
import torch.nn as nn

from math import pi, log, exp
from scipy.stats import norm

from typing import Callable, Type, Tuple

from utils import ScoreNetwork, m, sigma
from neural_sde import NeuralSDE
from plotting import plot_nsde_marginals

# -------------------------- Defining GMM, density, and score --------------------------
def q_GMM_batch(
    x: torch.Tensor, means: torch.Tensor, cov: torch.Tensor, mixture_props: torch.Tensor, eps: float = 1e-5,
):
    """
    Evaluates the density of a d-dimensional GMM on a tensor x of shape (batch_size, d)
    """

    cov_reg = cov + eps * torch.eye(x.shape[1], dtype=x.dtype, device=x.device) 
    sigma_inv = torch.linalg.inv(cov_reg)
    det_sigma = torch.det(cov_reg)
    norm_const = ((2 * pi) ** (-x.shape[1]/2)) * (det_sigma ** (-1/2))

    diff = x[:, None, :] - means[None, :, :]
    exp_term = torch.einsum('nkd,dd,nkd->nk', diff, sigma_inv, diff)
    component_pdf = norm_const * torch.exp(-0.5 * exp_term)
    pdf_values = component_pdf @ mixture_props
    return pdf_values


class GMM:
    """
    A class representing a Gaussian mixture model with specified means, mixture proportions, and common covariance
    """

    def __init__(
        self, means: torch.Tensor, cov: torch.Tensor, mixture_props: torch.Tensor
    ):
        super().__init__()
        self.means = means
        self.cov = cov
        self.mixture_props = mixture_props

    def sample(self, N: int):
        chol_factor = torch.linalg.cholesky(self.cov)
        idx = torch.multinomial(self.mixture_props, N, replacement=True)
        chosen_means = self.means[idx]
        base_samples = torch.randn(
            N, self.cov.shape[0], device=self.cov.device, dtype=self.cov.dtype
        )
        return base_samples @ chol_factor.T + chosen_means

    def density(self, x: torch.Tensor):
        return q_GMM_batch(x, self.means, self.cov, self.mixture_props)

    def density_convolved(self, x: torch.Tensor, m: torch.Tensor, S: torch.Tensor):
        """
        Compute the density of the GMM convolved with a Gaussian N(x|m, S) distribution
        """
        means_new = m * self.means
        cov_new = (m**2) * self.cov + S
        return q_GMM_batch(x, means_new, cov_new, self.mixture_props)


def gmm_score(x: torch.Tensor, t: torch.Tensor, gmm: GMM, m_fn: Callable[[torch.Tensor], torch.Tensor], sigma_fn: Callable[[torch.Tensor], torch.Tensor], eps: float = 1e-5):
    """
    Analytically computes the score of a Gaussian mixture model **with diagonal covariance**
    """
    n, d = x.shape
    t = t.reshape(-1)
    device, dtype = x.device, x.dtype

    # Compute mean and covariance of convolved density, which is again a GMM
    m_t = m_fn(t) # (N)
    sigma_t = sigma_fn(t) # (N)
    var0 = gmm.cov[0, 0]
    total_var = (m_t**2) * var0 + sigma_t**2 # (N)
    means_t = gmm.means * m_t[:, None, None] # (N, K, D)

    # Compute the score of the GMM
    log_w = torch.log(gmm.mixture_props) # (K)
    const_part = -0.5 * (d * torch.log(2 * pi * total_var)) # (N)
    diff = x[:, None, :] - means_t
    mahal_sq = diff.pow(2).sum(-1) # (N, K)
    a = log_w[None, :] + const_part[:, None] - 0.5 * (mahal_sq / total_var[:, None]) # (N, K)

    log_q = torch.logsumexp(a, dim=1, keepdim=True) 
    r = torch.exp(a - log_q)  # (N, K)

    numer = (r[..., None] * (means_t - x[:, None, :])).sum(1) # (N, D)
    score = numer / total_var[:, None] # (N, D)
    return score


def gmm_score_product(x: torch.Tensor, t: torch.Tensor, gmm_1d: GMM, m_fn: Callable[[torch.Tensor], torch.Tensor], sigma_fn: Callable[[torch.Tensor], torch.Tensor]):
    """
    Compute the score of a product measure, where each dimension is a Gaussian mixture model

    A wrapper around gmm_score
    """
    n, d = x.shape
    x_flat = x.reshape((-1, 1))  
    t = t.reshape((-1, 1))            
    t_flat = t.repeat(1, d).reshape(-1)
    score_1d = gmm_score(x_flat, t_flat, gmm_1d, m_fn, sigma_fn) 
    score = score_1d.reshape(n, d)
    return score

# -------------------------- Plotting --------------------------
def plot_marginals_and_gmm_density(
    model: NeuralSDE,
    gmm: GMM,
    Nsamps: int,
    ts: list = [0.0, 0.75, 0.9, 0.95, 0.99, 1.0],
    xlim=(-3, 3),
    m: Callable[[torch.Tensor], torch.Tensor] = m,
    sigma: Callable[[torch.Tensor], torch.Tensor] = sigma,
):

    fig, axs = plot_nsde_marginals(
        model, ts=ts, Nsamps=Nsamps, xlim=xlim, hist_kwargs={"color": "black"}
    )

    # Overlay densities of true backward process
    bins = torch.linspace(*xlim, 100).unsqueeze(-1)
    for i, t in enumerate(ts):
        row, col = divmod(i, 3)
        ax = axs[row, col] if hasattr(axs, "__getitem__") else axs[i]

        m_t = m(torch.tensor(1.0 - t))
        s_t = sigma(torch.tensor(1.0 - t))
        S_t = (s_t**2) * torch.eye(1)

        # Temporarily transfer the GMM onto CPU (and then move it back)
        device = gmm.means.device
        gmm.means, gmm.cov, gmm.mixture_props = gmm.means.cpu(), gmm.cov.cpu(), gmm.mixture_props.cpu()
        dens = gmm.density_convolved(bins, m_t, S_t).squeeze(-1)
        gmm.means, gmm.cov, gmm.mixture_props = gmm.means.to(device), gmm.cov.to(device), gmm.mixture_props.to(device)

        ax.plot(
            bins.detach(),
            dens.detach(),
            linewidth=2,
            label="true density",
            color="#A52A2A",
        )

    return fig, axs

# -------------------------- Score network for fine-tuning a GMM --------------------------
class ScoreOffsetNetwork(ScoreNetwork):
    """
    A score network which is equal to a "ground truth" score function plus a neural network offset
    The offset is initialized to zero
    """
 
    def __init__(
        self,
        analytical_score: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        input_dim: int,
        hidden_dim: int,
        time_emb_dim: int,
        sigma: Callable[[torch.Tensor], torch.Tensor],
        device: Type[torch.device] = torch.device("cpu"),
    ) -> None:

        super().__init__(input_dim, hidden_dim, time_emb_dim, sigma, device)

        self.analytical_score = analytical_score # analytical score function

        final_lin = self.net[-1] # initialize the weights of the final layer of the score network to zeros
        nn.init.zeros_(final_lin.weight)     
        nn.init.zeros_(final_lin.bias)

    def score(
        self, x: torch.Tensor, t: torch.Tensor, jitter: float = 1e-5
    ) -> torch.Tensor:
        
        analytical_score = self.analytical_score(x, t)
        score_offset = super().score(x, t, jitter)
        return analytical_score + score_offset

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(x, t)

# -------------------------- Analytical solution to max entropy problem --------------------------
def solve_max_ent_gmm(bprop: float, hstar: float, diff: float, std: float) -> Tuple[float, float]:
    """
    Computes the solution to the maximum entropy problem in a two mode GMM
    bprop * N(diff / 2, \sigma^2) + (1 - bprop) * N(-diff / 2, \sigma^2),     diff > 0
    with constraint function h(x) = 1{x > 0}

    NOTE: bprop is the mixture proportion, not P(X > 0)
    """

    pb = bprop * norm.cdf(diff / (2 * std)) + (1 - bprop) * (1 - norm.cdf(diff / (2 * std))) # probability >= 0 under the GMM
    alpha_star = log((hstar * (1-pb)) / ((1 - hstar) * pb) ) # true alpha star is the log odds ratio
    Z = (1-pb) + exp(alpha_star) * pb # true normalizing constant is just (1-h_b)/(1-h^*)
    min_kl = alpha_star * hstar - log(Z)
    return alpha_star, min_kl