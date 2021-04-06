"""
This file implements the base agent class.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import TypeVar

# Third-party dependencies
import torch

# Type definitions
Action = TypeVar("Action")
# Need this for type hints, true base
# class is hidden.
Scheduler = TypeVar("Scheduler")


class Agent(ABC):
    """
    Base agent class. All other agents inherit
    from this class.
    """
    def __init__(self, tensor_specs: dict):
        """
        Ctor.
        :param tensor_specs: A dictionary with optional
        pytorch tensor parameters. Keys can be for example 'dtype' and
        'device'.
        """
        self.tensor_specs = tensor_specs

    def tensor(self, *args) -> torch.Tensor:
        """
        Utility function for creating a torch tensor. It will always
        append the arguments from tensor_specs to the parameters, so
        it is not needed to specify them every time you create a new
        tensor.
        :param args: Arguments to the tensor.
        :return: New pytorch tensor.
        """
        return torch.tensor(*args, **self.tensor_specs)

    @abstractmethod
    def train_step(self, *args) -> Action:
        """
        Step function called when training. Returns an action and
        (potentially) updates its parameters.
        :return: An action to take.
        """

    @abstractmethod
    def __call__(self, *args) -> Action:
        """
        Step function in inference mode. Returns an action but does not
        train.
        :return: An action to take.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Saves the current parameters of the agent
        to the specified file.
        :param path: Path to output file.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads agent from a saved checkpoint.
        :param path: Path to the checkpoint file.
        """
