import os
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.distributions import Normal

from cgm.cgm import calibrate_reward, calibrate_relaxed, calibrate_forward_kl
from cgm.model import Model
from cgm.utils import DictLogger, BestCheckpoint, CheckpointEveryN, load_checkpoint


class GaussianModel(Model[torch.Tensor]):
    def __init__(self):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1))
        self.dist = Normal(self.mu, torch.ones(1))

    def sample(self, N: int) -> torch.Tensor:
        return self.dist.sample((N,))

    def log_p(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.dist.log_prob(x).squeeze()


def test_calibrate_relax():
    torch.manual_seed(11)
    base = GaussianModel()
    hstar = torch.ones(1)
    logger = DictLogger()
    epochs = 5
    finetuned = calibrate_relaxed(
        base,
        h=nn.Identity(),
        hstar=hstar,
        lambd=1.0,
        epochs=epochs,
        batch_size=64,
        logger=logger,
        optimizer_params={"lr": 1e-1},
    )
    assert finetuned.mu.item() > 0  # modify finetuned model in expected direction
    assert set(map(len, logger.metrics.values())) == {epochs}  # check logging correctly


def test_calibrate_reward():
    torch.manual_seed(11)
    base = GaussianModel()
    hstar = torch.ones(1)
    logger = DictLogger()
    epochs = 5
    finetuned = calibrate_reward(
        base,
        h=nn.Identity(),
        N_samp=128,
        hstar=hstar,
        epochs=epochs,
        batch_size=64,
        logger=logger,
        optimizer_params={"lr": 1e-1},
    )
    assert finetuned.mu.item() > 0  # modify finetuned model in expected direction
    assert set(map(len, logger.metrics.values())) == {epochs}  # check logging correctly


def test_calibrate_forward_kl():
    torch.manual_seed(11)
    base = GaussianModel()
    hstar = torch.ones(1)
    logger = DictLogger()
    epochs = 5
    finetuned = calibrate_forward_kl(
        base,
        h=nn.Identity(),
        N_samp=128,
        hstar=hstar,
        epochs=epochs,
        batch_size=64,
        logger=logger,
        optimizer_params={"lr": 1e-1},
    )
    assert finetuned.mu.item() > 0  # modify finetuned model in expected direction
    assert set(map(len, logger.metrics.values())) == {epochs}  # check logging correctly


def test_best_checkpoint():
    torch.manual_seed(11)
    base = GaussianModel()
    hstar = torch.ones(1)
    logger = DictLogger()
    epochs = 1
    with TemporaryDirectory() as tempdir:
        ckp_fn = BestCheckpoint(tempdir)
        finetuned = calibrate_relaxed(
            base,
            h=nn.Identity(),
            hstar=hstar,
            lambd=1.0,
            epochs=epochs,
            batch_size=64,
            logger=logger,
            optimizer_params={"lr": 1e-1},
            checkpoint_fn=ckp_fn,
        )
        ckpt_path = os.path.join(tempdir, ckp_fn.checkpoint_name())
        test_model = GaussianModel()
        next_epoch = load_checkpoint(test_model, ckpt_path)
        assert (
            next_epoch == 1
        )  # check that first epoch is checkpointed as best, since only one epoch
        assert (
            test_model.state_dict() == finetuned.state_dict()
        )  # check that parameters correctly loaded


def test_checkpoint_every_n():
    torch.manual_seed(11)
    base = GaussianModel()
    hstar = torch.ones(1)
    logger = DictLogger()
    epochs = 6
    with TemporaryDirectory() as tempdir:
        ckp_fn = CheckpointEveryN(tempdir, N=2)
        finetuned = calibrate_relaxed(
            base,
            h=nn.Identity(),
            hstar=hstar,
            lambd=1.0,
            epochs=epochs,
            batch_size=64,
            logger=logger,
            optimizer_params={"lr": 1e-1},
            checkpoint_fn=ckp_fn,
        )
        for step in range(epochs):
            if (step + 1) % ckp_fn.N == 0:
                ckpt_path = os.path.join(
                    tempdir, ckp_fn.checkpoint_name(loss=None, epoch=step)
                )
                assert os.path.exists(ckpt_path)
        # check last checkpoint
        test_model = GaussianModel()
        next_epoch = load_checkpoint(test_model, ckpt_path)
        assert (
            next_epoch == epochs
        )  # check that first epoch is checkpointed as best, since only one epoch
        assert (
            test_model.state_dict() == finetuned.state_dict()
        )  # check that parameters correctly loaded
