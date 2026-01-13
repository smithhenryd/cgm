import torch
import torch.nn as nn

from pathlib import Path

from ml_tarflow.transformer_flow import Model as TarflowModel

from typing import Any, Optional, Union
from cgm.model import module_device


def load_model(model: TarflowModel, path: Union[str, Path]):
    try:
        model.load_state_dict(
            torch.load(path, map_location=module_device(model), weights_only=True)
        )
        print(f"Successfully loaded model at {path} on device {module_device(model)}")
    except:
        raise ValueError("No checkpoint file")


class TarflowMap(nn.Module):
    """
    Class representing the normalizing flow map defined by TarFlow
    """

    def __init__(
        self,
        path: Union[str, Path],
        device: torch.device,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        num_classes: int,
        noise_std: float,
        sample_cls: Optional[int] = None,
        sampling_args: Optional[dict[str, Any]] = {},
    ):

        super().__init__()

        self.img_size, self.patch_size, self.in_channels, self.noise_std = (
            img_size,
            patch_size,
            in_channels,
            noise_std,
        )

        if sample_cls is not None:
            assert (
                sample_cls < num_classes
            ), f"sample_cls ({sample_cls}) must be less than num_classes ({num_classes})"

        # Load pre-trained model
        self.model = TarflowModel(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            channels=channels,
            num_blocks=num_blocks,
            layers_per_block=layers_per_block,
            num_classes=num_classes,
        ).to(device)
        load_model(self.model, path)  # load model

        # For sampling
        self.sampling_args = sampling_args  # optional keyword arguments for sampling
        self.sample_cls = sample_cls
        self.lr = self.img_size**2 * self.in_channels * self.noise_std**2

    @property
    def device(self) -> torch.device:
        return module_device(self.model)

    def reverse(self, zs: torch.Tensor, perform_tweedie: bool = False) -> torch.Tensor:
        """
        Given z from N(0, I), computes samples x = f^{-1}(z)

        NOTE: Set perform_tweedie = True for FID evaluation, False for training
        """
        zs = zs.reshape(
            (
                -1,
                (self.img_size // self.patch_size) ** 2,
                self.in_channels * self.patch_size**2,
            )
        )
        batch_size = zs.shape[0]

        chunk_size = min(batch_size, 256)
        y = (
            torch.full(
                (batch_size,), self.sample_cls, dtype=torch.int, device=self.device
            )
            if self.sample_cls is not None
            else None
        )
        for i in range(0, batch_size, chunk_size):
            j = min(i + chunk_size, batch_size)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                samps_i = self.model.reverse(x=zs[i:j], y=y[i:j], **self.sampling_args)
            samps = samps_i if i == 0 else torch.concat([samps, samps_i], dim=0)

        # Then perform the final denoising step using Tweedie's formula
        if perform_tweedie:
            ssamps = self._tweedie_step_microbatched(  # chunk computation since we are backpropogating through the network
                samps=samps,
                chunk_size=min(batch_size, 20),
                amp_dtype=torch.bfloat16,
            )
            eps = (samps - ssamps).reshape((batch_size, -1))
            samps = ssamps.reshape((batch_size, -1))
        else:
            samps = samps.reshape((batch_size, -1))
            eps = torch.zeros_like(samps, device=self.device)
        return (
            samps,
            eps,
        )  # store both the original samples and the Tweedie denoised samples

    def _tweedie_step_microbatched(
        self,
        samps: torch.Tensor,
        chunk_size: int = 10,
        amp_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Applies Tweedie's formula to x = f^{-1}(z) to compute denoised samples
        """

        batch_size = samps.shape[0]
        outs = []
        for i in range(0, batch_size, chunk_size):
            j = min(i + chunk_size, batch_size)
            bsz = j - i

            with torch.enable_grad():
                x = samps[i:j].detach().requires_grad_(True)
                with torch.autocast(
                    device_type=self.device.type, dtype=amp_dtype, cache_enabled=False
                ):
                    z, outputs, logdets = self.model(
                        x,
                        y=torch.tensor([self.sample_cls], dtype=torch.int).repeat(bsz),
                    )
                    loss = self.model.get_loss(z, logdets)
                grad = torch.autograd.grad(
                    loss, [x], create_graph=False, retain_graph=False
                )[0]

            x = (x - (bsz * self.lr) * grad).detach()
            outs.append(x)
            # Help allocator between chunks
            del x, z, outputs, logdets, grad, loss
            torch.cuda.empty_cache()

        return torch.cat(outs, dim=0)

    def log_p(self, xs: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Given samples x, compute their log probabilities logp(x)

        NOTE: in Tarflow, independent noise is first added to the samples
        """
        xs, eps = xs.reshape(
            (-1, self.in_channels, self.img_size, self.img_size)
        ), eps.reshape((-1, self.in_channels, self.img_size, self.img_size))
        batch_size = xs.shape[0]
        xs = xs + eps

        zs, _, logdets = self.model(
            xs, y=torch.tensor([self.sample_cls], dtype=torch.int).repeat(batch_size)
        )  # (batch_size, dim), (batch_size)
        logdets *= (
            self.img_size
        ) ** 2 * self.in_channels # log determinants are averaged over the last two dimensions
        return -(1 / 2) * (zs**2).sum(axis=[-2, -1]) + logdets  # (batch_size)

    def forward(self, xs: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        Given samples x, compute noise z = f(x)
        """
        xs, eps = xs.reshape(
            (-1, self.in_channels, self.img_size, self.img_size)
        ), eps.reshape((-1, self.in_channels, self.img_size, self.img_size))
        batch_size = xs.shape[0]
        xs = xs + eps

        zs, _, _ = self.model(
            xs, y=torch.tensor([self.sample_cls], dtype=torch.int).repeat(batch_size)
        )  # then evaluate the model
        return zs
