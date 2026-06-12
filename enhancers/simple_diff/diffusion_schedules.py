from abc import abstractmethod
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical, Exponential, OneHotCategorical
from tqdm import tqdm


class DiffusionSchedule(nn.Module):
    @abstractmethod
    def alpha_t(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the fraction of the original signal remaining at time t
        t: [bsz]
        output shape should be [bsz, 1] or [bsz, seq_len]
        """
        raise NotImplementedError("alpha_t method must be implemented")

    @abstractmethod
    def deriv(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the derivative of alpha_t(t) w.r.t. t
        """
        raise NotImplementedError("deriv method must be implemented")

    def cross_entropy_loss_weight(self, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        the weight for the cross entropy loss at time t
        calculated as -1 * d/dt alpha_t(t) / (1 - alpha_t(t))
        """
        return -self.deriv(t, **kwargs) / (1 - self.alpha_t(t, **kwargs))

    def integrated_rate_solve(
        self, target: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError("No integrated_rate_solve method implemented")

    @property
    def learned(self):
        if self.parameters():
            return True
        return False

    def p_t(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        vocab_size: int = None,
        **alpha_t_kwargs,
    ) -> Categorical:
        """
        x.shape = [bsz, seq_len, vocab_size]
        t.shape = [bsz] (or float scalar)
        """
        if x_0.ndim == 2:
            assert (
                vocab_size is not None
            ), "vocab_size must be provided for 2D (categorical) input"
            bsz, seq_len = x_0.shape
            x_0 = F.one_hot(x_0, num_classes=vocab_size)
        elif x_0.ndim == 3:
            bsz, seq_len, vocab_size = x_0.shape
        else:
            raise ValueError(f"{x_0.ndim=}. x_0 should have 2 or 3 dimensions")
        if type(t) == float:
            t = torch.full((bsz,), t, device=x_0.device)
        mu_t = torch.zeros((bsz, seq_len, vocab_size + 1), device=x_0.device)
        alpha_t = self.alpha_t(t, **alpha_t_kwargs).detach()
        # TODO: the following could be simplified by using a one-hot to represent the mask
        # signal component
        mu_t[..., :-1] = x_0 * alpha_t[..., None]
        # noise component
        mu_t[..., -1] += 1 - alpha_t
        return Categorical(probs=mu_t)

    def q_s(
        self,
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        s: float,
        t: float,
        **alpha_t_kwargs,
    ) -> Categorical:
        """
        x_t.shape = [bsz, seq_len] (categorical)
        x_0.shape = [bsz, seq_len, vocab_size]

        # based on section: "Time reversal of the forward process given x0"
        of https://arxiv.org/abs/2406.04329
        "Simplified and Generalized Masked Diffusion for Discrete Data"
        or eq (6) of https://arxiv.org/abs/2406.07524
        "Simple and Effective Masked Diffusion Language Models
        """
        bsz, seq_len, vocab_size = x_0.shape
        augmented_vocab_size = vocab_size + 1
        x_t = F.one_hot(x_t, num_classes=augmented_vocab_size)

        alpha_s = self.alpha_t(
            torch.tensor([s], dtype=torch.float64, device=x_t.device), **alpha_t_kwargs
        )
        alpha_t = self.alpha_t(
            torch.tensor([t], dtype=torch.float64, device=x_t.device), **alpha_t_kwargs
        )

        # mask / absorbing state
        m = torch.zeros(
            (bsz, seq_len, augmented_vocab_size),
            device=x_t.device,
            dtype=torch.float64,
        )
        m[..., -1] = 1

        x_0_full = torch.zeros_like(m)
        x_0_full[..., :-1] = x_0

        # TODO: the following could be simplified by using a one-hot to represent the mask
        unmask_prob = ((alpha_s - alpha_t) / (1 - alpha_t)).unsqueeze(-1)
        stay_mask_prob = 1 - unmask_prob
        masked_token_probs = (stay_mask_prob * m) + unmask_prob * x_0_full

        mu_s = torch.where(
            x_t[..., -1:] == 1,  # mask state
            masked_token_probs,
            x_t,
        )

        return Categorical(probs=mu_s)

    def log_q_s(
        self,
        x_t: torch.Tensor,
        log_x_0: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_t.shape = [bsz, seq_len] (categorical)
        log_x_0.shape = [bsz, seq_len, vocab_size]
        t.shape = [bsz]
        """
        bsz, L, vocab_size = log_x_0.shape
        alpha_s = self.alpha_t(s)
        alpha_t = self.alpha_t(t)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        stay_mask_prob = 1 - unmask_prob

        # log probs for positions staying unmasked
        unmask_pos_log_probs = torch.full(
            (bsz, L, vocab_size + 1),
            fill_value=-1e9,
            device=log_x_0.device,
        )
        unmask_pos_log_probs = torch.scatter(
            unmask_pos_log_probs,
            index=x_t.unsqueeze(-1),
            dim=-1,
            src=torch.ones_like(unmask_pos_log_probs),
        )
        unmask_pos_log_probs = unmask_pos_log_probs.log_softmax(dim=-1)

        # logits for unmasking
        stay_mask_tensor = stay_mask_prob.log().repeat(1, L)  # bsz x L
        unmask_tensor = log_x_0 + unmask_prob.log().view(bsz, 1, 1)
        log_prob_tensor = torch.cat(
            [unmask_tensor, stay_mask_tensor.unsqueeze(-1)],
            dim=-1,
        )

        # combine
        masked = x_t == vocab_size  # bsz x L
        log_prob_tensor = torch.where(
            masked.unsqueeze(-1),
            log_prob_tensor,
            unmask_pos_log_probs,
        )
        return log_prob_tensor

    @torch.no_grad()
    def sample_tau_leap(
        self,
        x_t: torch.Tensor,
        mu_0_model: nn.Module,
        delta_t: float,
        mu_0_kwargs: dict = None,
        cnn_mode: bool = False,
        min_t: float = 0,
        verbose: bool = False,
        return_intermediates: bool = False,
        **alpha_t_kwargs,
    ) -> torch.Tensor:
        """
        mu_0_model predicts the initial (time 0) token probabilities
        mu_0_kwargs are the keyword arguments to pass to mu_0_model.forward besides x_t
        """
        mu_0_kwargs = mu_0_kwargs or {}
        intermediates = [x_t.clone().cpu()]
        for t in tqdm(
            torch.arange(1, delta_t + min_t, -delta_t),
            desc="Backwards sampling",
            disable=not verbose,
        ):
            logits = mu_0_model(x_t, **mu_0_kwargs)
            if cnn_mode:
                logits = logits.swapaxes(1, 2)
            x_0 = F.softmax(logits, dim=-1)
            q_s = self.q_s(
                x_t=x_t, x_0=x_0, s=t.item() - delta_t, t=t.item(), **alpha_t_kwargs
            )
            x_t = q_s.sample()
            if return_intermediates:
                intermediates.append(x_t.clone().cpu())
        # resample if any tokens are still masked (necessary for larger delta_t)
        x_t_masked = x_t == x_0.shape[-1]
        mask_mean = x_t_masked.float().mean()
        if mask_mean > 0:
            print(f"{mask_mean:.1%} still masked after backwards sampling")
            resample = x_0.argmax(-1)
            x_t = torch.where(x_t_masked, resample, x_t)
        if return_intermediates:
            intermediates = torch.stack(intermediates, dim=1)
            return x_t, intermediates
        return x_t

    def sample_gillespie(
        self,
        x_t: torch.Tensor,
        mu_0_model: nn.Module,
        mu_0_kwargs: dict = None,
        cnn_mode: bool = False,
        temperature: float = 1.0,
        verbose: bool = False,
        return_intermediates: bool = False,
        **alpha_t_kwargs,
    ) -> torch.Tensor:
        x_t = x_t.clone()
        mu_0_kwargs = mu_0_kwargs or {}
        bsz, seq_len = x_t.shape
        t = torch.ones(bsz, device=x_t.device) - 1e-3
        ts = [t.clone().cpu()]
        intermediates = [x_t.clone().cpu()]
        batch_range = torch.arange(bsz, device=x_t.device)
        for _ in tqdm(
            range(seq_len),
            desc=f"gillespi sampling with temp. {temperature:.2f}",
            disable=not verbose,
        ):
            logits = mu_0_model(x_t, **mu_0_kwargs) / temperature
            if cnn_mode:
                logits = logits.swapaxes(1, 2)
            still_masked = x_t == logits.shape[-1]

            # sample unmask time
            # done by summing the rates and sampling one exponential
            # then sampling which position (assuming all positions have equal rates)
            if return_intermediates:
                exp_target = Exponential(
                    torch.ones_like(x_t, dtype=torch.float32)
                ).sample()
                delta_t = self.integrated_rate_solve(
                    exp_target, t.unsqueeze(1), **alpha_t_kwargs
                )
                # don't unmask already unmasked token
                delta_t = torch.where(
                    still_masked,
                    delta_t,
                    torch.inf,
                )
                which_pos = delta_t.argmin(dim=1, keepdim=True)
                delta_t = torch.gather(delta_t, index=which_pos, dim=1).squeeze()
                t = t - delta_t
                ts.append(t.clone().cpu())
                which_pos = which_pos.squeeze()
            else:
                which_pos = Categorical(still_masked).sample()
            # sample next token
            token_logits = logits[batch_range, which_pos]  # bsz x vocab size
            which_token = Categorical(token_logits.softmax(dim=-1)).sample()
            x_t[batch_range, which_pos] = which_token
            if return_intermediates:
                intermediates.append(x_t.clone().cpu())
        if return_intermediates:
            intermediates = torch.stack(intermediates, dim=1)
            return intermediates, torch.stack(ts, dim=1)
        return x_t


class CosineDiffusionSchedule(DiffusionSchedule):
    @staticmethod
    def alpha_t(t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.cos(math.pi / 2 * (1 - t)).unsqueeze(1)

    @staticmethod
    def cross_entropy_loss_weight(t: torch.Tensor) -> torch.Tensor:
        # Table 4 of https://arxiv.org/abs/2406.04329
        return (
            (math.pi / 2 * torch.tan(math.pi / 2 * (1 - t)))
            .clamp(min=1e-2, max=1e2)
            .unsqueeze(1)
        )

    @staticmethod
    def deriv(t: torch.Tensor) -> torch.Tensor:
        return -math.pi / 2 * torch.sin(math.pi / 2 * (1 - t)).unsqueeze(1)

    @staticmethod
    def integrated_rate_solve(target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Solve for t2 < t1 such that
        ∫ rate(τ) dτ from t2 to t1 == target
        where rate(t)= (π/2)*tan(π/2*(1 - t)).
        """
        # forward integral form:
        #  target = ln(cos(π/2(1 - t2))) - ln(cos(π/2(1 - t1)))
        #  => cos(π/2(1 - t2)) = cos(π/2(1 - t1))*exp(-target)
        arccos_input = torch.cos((math.pi / 2) * (1 - t)) * torch.exp(-target)
        return 1 - 2 / math.pi * torch.arccos(arccos_input)
