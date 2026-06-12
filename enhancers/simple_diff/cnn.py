from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class CNNLayerNorm1D(nn.Module):
    """
    ConvNeXt-style LayerNorm for 1D convs:
    - input: (N, C, L)
    - normalize over C for each (N, L)
    """

    def __init__(self, n_filters: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_filters, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_filters, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat


class CNNBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        dilation: int,
        cond_emb_dim: int | None = None,
        sequence_conditioning: bool = False,
        dropout: float = 0.0,
        kernel_size: int = 9,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            CNNLayerNorm1D(n_filters),
            nn.Conv1d(
                n_filters,
                n_filters,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout1d(dropout),
        )
        self.is_conditional = cond_emb_dim is not None
        self.sequence_conditioning = sequence_conditioning
        if self.is_conditional:
            if sequence_conditioning:  # conditioning is sequence length x embed_dim
                self.cond_layer = nn.Sequential(
                    CNNLayerNorm1D(cond_emb_dim),
                    nn.Conv1d(cond_emb_dim, n_filters, kernel_size=1, padding="same"),
                )
            else:  # conditioning is single embedding
                self.cond_layer = nn.Sequential(
                    nn.LayerNorm(cond_emb_dim),
                    nn.Linear(cond_emb_dim, n_filters),
                )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.is_conditional:
            z = self.cond_layer(cond)
            if not self.sequence_conditioning:
                z = z.unsqueeze(-1)  # bsz x d -> bsz x d x 1
            h = x + z
        else:
            h = x
        h = self.conv(h)
        return x + h  # residual


class CNNBody(nn.Module):
    def __init__(
        self,
        n_filters: int,
        num_dilation_blocks: int,
        cond_emb_dim: int | None = None,
        sequence_conditioning: bool = False,
        dropout: float = 0,
        kernel_size: int = 9,
    ):
        super().__init__()
        dilations = [1, 1, 4, 16, 64]
        self.blocks = nn.ModuleList()
        for _ in range(num_dilation_blocks):
            for i, dilation in enumerate(dilations):
                self.blocks.append(
                    CNNBlock(
                        n_filters,
                        dilation=dilation,
                        cond_emb_dim=(
                            cond_emb_dim if i == 0 else None
                        ),  # condition only at start of block
                        sequence_conditioning=sequence_conditioning,
                        dropout=dropout,
                        kernel_size=kernel_size,
                    )
                )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return x


@dataclass
class CNNConfig:
    input_channels: int = 5
    output_channels: int = 4
    n_filters: int = 256
    num_dilation_blocks: int = 2
    cond_dim: int | None = None
    sequence_conditioning: bool = False
    dropout: float = 0
    kernel_size: int = 9
    final_kernel_size: int = 1

    @property
    def is_conditional(self) -> bool:
        return self.cond_dim is not None


class CNN(nn.Module):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.input_conv = nn.Conv1d(
            cfg.input_channels,
            cfg.n_filters,
            kernel_size=cfg.kernel_size,
            padding="same",
        )
        if self.cfg.is_conditional:
            if cfg.sequence_conditioning:
                self.cond_encoder = nn.Sequential(
                    nn.Conv1d(
                        cfg.cond_dim, cfg.n_filters, cfg.kernel_size, padding="same"
                    ),
                    CNNBody(
                        n_filters=cfg.n_filters,
                        num_dilation_blocks=1,
                        cond_emb_dim=None,
                        dropout=cfg.dropout,
                        kernel_size=cfg.kernel_size,
                    ),
                )
            else:
                self.cond_encoder = nn.Sequential(
                    nn.LayerNorm(cfg.cond_dim),
                    nn.Linear(cfg.cond_dim, cfg.n_filters),
                    nn.ReLU(),
                    nn.Linear(cfg.n_filters, cfg.n_filters),
                )
        self.body = CNNBody(
            n_filters=cfg.n_filters,
            num_dilation_blocks=cfg.num_dilation_blocks,
            cond_emb_dim=cfg.n_filters if self.cfg.is_conditional else None,
            sequence_conditioning=cfg.sequence_conditioning,
            dropout=cfg.dropout,
            kernel_size=cfg.kernel_size,
        )
        self.output_conv = nn.Conv1d(
            cfg.n_filters,
            cfg.output_channels,
            kernel_size=cfg.final_kernel_size,
            padding="same",
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_conv(x)
        if self.cfg.is_conditional:
            cond = cond.float()
            cond = self.cond_encoder(cond)
        x = self.body(x, cond)
        return self.output_conv(x)
