from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from cgm.utils import module_device

import torch
import torch.nn as nn


SampleType = TypeVar("SampleType")


class Model(Generic[SampleType], nn.Module, ABC):
    """
    Abstract base class representing the generative model to be calibrated
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, N: int) -> SampleType:
        """
        Draws N independent samples from the generative model

        N: the number of samples to draw from the generative model
        """
        raise NotImplementedError

    @abstractmethod
    def log_p(self, x: SampleType, **kwargs) -> torch.Tensor:
        """
        Computes the log density of the generative model, evaluated at samples x

        x: samples on which to evaluate the log probability
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return module_device(self)
