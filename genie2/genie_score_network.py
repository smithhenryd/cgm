import torch
import torch.nn as nn

import os

from genie.sampler.unconditional import UnconditionalSampler
from genie.utils.model_io import load_pretrained_model


def genie_denoiser(
    x: torch.tensor, t: int, genie_model: UnconditionalSampler
) -> torch.tensor:
    """
    The Genie denoiser, predicts the noise from the structure and the timestep

    x: a shape (batch_size, D) tensor, where D / 3 is the number of backbone atoms
    t: an integer timestep in [1, 10^3] indicating the denoising step,
    with timestep 10^3 representing standard Gaussian noise and timestep 0 representing denoised samples
    """
    assert x.dim() == 2, "Expected input of shape (batch_size, D)"
    B, N = x.shape[0], x.shape[1] // 3  # Number of atoms

    if len(t.shape) == 2:
        assert t.shape[1] == 1
        t = t[:, 0]

    # Arguments for Genie2 call
    params = {
        "length": N,
        "num_samples": B,
        "atom_pos": x.reshape([B, N, 3]),
        "step": t,
    }

    eps_hat = genie_model.predict_noise(params)
    eps_hat = eps_hat.reshape([x.shape[0], -1])

    return eps_hat


def get_genie_model(device: torch.device) -> UnconditionalSampler:
    """
    Loads the Genie2 model onto the specified device
    """

    model = load_pretrained_model("genie2/results", "base", "40", device).eval()

    # Load sampler
    sampler = UnconditionalSampler(model)
    return sampler


class GenieScoreNetwork(nn.Module):
    """
    The score network corresponding to the Genie model
    """

    def __init__(
        self, device: torch.device, noise_scale: float = 1.0, pdb_dir: str = None
    ):

        super(GenieScoreNetwork, self).__init__()

        # Store Genie2 model and sampler
        sampler = get_genie_model(
            device
        )  # NOTE: the sampler itself is NOT the Genie2 model, see sampler/base.py
        self.denoiser = (
            sampler.model
        )  # This ensures the parameters of the Genie2 model are properly recognized by Torch; see diffusion/genie.py
        sampler.model = self.denoiser
        self.sampler = sampler

        self.noise_scale = noise_scale
        self.pdb_dir = pdb_dir

    def score(self, x: torch.Tensor, t: float) -> torch.Tensor:
        # Map t to nearest integer time t_discrete for Genie
        t_discrete = torch.clip((t * 1000).to(torch.long), min=1, max=None)

        eps_hat = genie_denoiser(x, t_discrete, self.sampler)

        # Conditional score is - eps / sqrt(Var[xt | x0])
        score = -eps_hat / self.denoiser.sqrt_one_minus_alphas_cumprod[t_discrete]

        # Save intermediate predictions in the backward process
        if self.pdb_dir is not None:
            os.makedirs(self.pdb_dir + "/frames/", exist_ok=True)

            # Compute prediction of x(0) from eps
            posterior_mean_coef1 = 1 / self.denoiser.sqrt_alphas_cumprod[t_discrete]
            posterior_mean_coef2 = (
                self.denoiser.sqrt_one_minus_alphas_cumprod[t_discrete]
                / self.denoiser.sqrt_alphas_cumprod[t_discrete]
            )
            x0_given_xt = x * posterior_mean_coef1 - eps_hat * posterior_mean_coef2
            x0_hat = x0_given_xt

            t_print = int(t_discrete[0].cpu())
            secstruct.save_pdb(
                x0_hat[0].cpu().numpy().reshape([-1, 3]),
                self.pdb_dir + f"frames/sample_denoise_x0_pred_t={t_print:04d}.pdb",
            )
            secstruct.save_pdb(
                x[0].cpu().numpy().reshape([-1, 3]),
                self.pdb_dir + f"frames/sample_denoise_xt_t={t_print:04d}.pdb",
            )
        return score / (
            self.noise_scale**2 * self.denoiser.sqrt_alphas[t_discrete]
        )  # Divide the score by (noise scale)^2 * \sqrt{\alpha(t)}

    def forward(self, x, t) -> torch.Tensor:
        return self.score(x, t)


def diffusion(
    t: torch.Tensor, sqrt_betas: torch.Tensor, noise_scale: float = 1.0
) -> torch.Tensor:
    """
    Diffusion coefficient that is used to define the Genie2 DDPM
    """
    t_discrete = torch.clip((t * 1000).to(torch.long), min=1, max=None)
    sqrt_beta = sqrt_betas[t_discrete]

    return (
        noise_scale * torch.sqrt(torch.tensor(1000.0, device=t.device)) * sqrt_beta
    )  # since noise is scaled by \sqrt{\delta t}


def drift(x: torch.Tensor, t: torch.Tensor, sqrt_alphas: torch.Tensor) -> torch.Tensor:
    """
    Drift function that is used to define the Genie2 DDPM
    """
    t_discrete = torch.clip((t * 1000).to(torch.long), min=1, max=None)
    sqrt_alpha_t = sqrt_alphas[t_discrete]

    return (
        -1000.0 * ((1.0 - sqrt_alpha_t[:, None]) / sqrt_alpha_t[:, None]) * x
    )  # Euler-Maruyama update is x_t + \delta t f(x_t, t), so be sure subtract off an extra factor of x_t
    # Also get a multiplicative factor of -1 since this is the drift of the forward process
