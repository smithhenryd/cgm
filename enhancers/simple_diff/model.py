from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributions
import torch.nn.functional as F

from simple_diff.utils import BaseLightning
from simple_diff.diffusion_schedules import CosineDiffusionSchedule
from simple_diff.cnn import CNNConfig, CNN


@dataclass
class CNNConditionalMDLMConfig:
    cnn_config: CNNConfig
    lr: float = 5e-4
    wd: float = 0.0
    print_freq: int = 1


class CNNConditionalMDLM(BaseLightning):
    """
    conditional CNN discrete diffusion model
    """

    def __init__(self, cfg: CNNConditionalMDLMConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.cnn = CNN(cfg.cnn_config)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.mask_token = cfg.cnn_config.input_channels - 1
        self.schedule = CosineDiffusionSchedule()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        cnn_output_shape: bool = False,
    ) -> torch.Tensor:
        """
        Expects x to be token IDs with shape bsz x seq_len.
        return: bsz x channels x seq_len if cnn_output_shape, otherwise bsz x seq_len x channels
        """
        x = F.one_hot(x, num_classes=self.cfg.cnn_config.input_channels)
        x = x.swapaxes(1, 2).float().contiguous()
        if (
            self.cfg.cnn_config.is_conditional
            and self.cfg.cnn_config.sequence_conditioning
        ):
            cond = cond.swapaxes(1, 2).contiguous()
        x = self.cnn(x, cond)
        if cnn_output_shape:
            return x
        return x.swapaxes(1, 2).contiguous()

    def get_loss(self, batch, prefix: str) -> torch.Tensor:
        if self.cfg.cnn_config.is_conditional:
            x_0, cond = batch
        else:
            x_0 = batch
            cond = None
        x_0 = x_0.to(torch.long)
        bsz, L = x_0.shape
        with torch.no_grad():
            # low discrepency sampling of times
            t = torch.linspace(0, 1, bsz).to(x_0.device)
            t_offset = torch.rand(1, device=x_0.device)
            t = (t + t_offset) % 1
            x_t = self.schedule.p_t(x_0, t, vocab_size=self.mask_token).sample()
            x_t_is_masked = x_t == self.mask_token
        logits = self(x_t, cond, cnn_output_shape=True)  # bsz x channels x seq_len
        loss_weight = self.schedule.cross_entropy_loss_weight(t)
        unmasked_loss = self.loss_fn(logits, x_0)  # bsz x seq_len
        batch_loss = (x_t_is_masked * unmasked_loss * loss_weight).sum(1)
        loss = batch_loss.mean() / x_t.size(1)
        self.smart_log("loss", loss, prefix)
        return loss

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seq_len: int,
        cond: torch.Tensor | None = None,
        **sample_kwargs,
    ) -> torch.Tensor:
        xT = torch.full((n, seq_len), self.mask_token, device=self.device)
        if cond is not None:
            mu_0_kwargs = {"cond": cond.to(self.device)}
        else:
            mu_0_kwargs = {}
        return self.schedule.sample_gillespie(
            xT, mu_0_model=self, mu_0_kwargs=mu_0_kwargs, **sample_kwargs
        )
