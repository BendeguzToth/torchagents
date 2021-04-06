"""
Implementation of double deep Q-learning.
"""

# Standard libraries
from typing import Optional, TypeVar, Callable
from collections import deque
import copy
import random
import logging

# Third-party dependencies
import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Optimizer

# Project files
from agent import Agent, Scheduler


class DDQN(Agent):
    """
    Double deep Q-learning with experience replays
    and with frozen target network.
    """
    def __init__(
            self,
            value_function: Module,
            optimizer:      Optimizer,
            gamma:          float,
            epsilon_fn:     Callable[[int], float],
            replay_buffer_size: int,
            replay_batch_size:  int,
            start_training_at:  int,
            unfreeze_freq:      int,
            device:             torch.device,
            lr_scheduler:       Optional[Scheduler] = None,
            verbose:            bool = False,
            floating_dtype:     torch.dtype = torch.float32
    ):
        """
        Ctor.
        :param value_function: A torch module that takes an input of shape
        (replay_batch_size, *state.shape) where state.shape is the shape of the
        state representation that is fed to the agent. The output should be of shape
        (replay_batch_size, |action|) where |action| is the number of different
        possible actions.
        :param optimizer: Optimizer to use for training.
        :param lr_scheduler: [Optional] Learning rate scheduler. Can be None.
        :param gamma: Reward discount rate. [0, 1]
        :param epsilon_fn: A function that as parameter gets the current
        step number (int) and returns an epsilon (float). Epsilon
        is the chance that a suboptimal action will be chosen in order
        to promote better exploration during training. The returned value
        needs to be in the interval [0, 1].
        :param replay_buffer_size: Size of the replay buffer.
        :param replay_batch_size: Batch size used when training from the replay
        buffer.
        :param start_training_at: Start training after at least this amount of data
        in experience buffer.
        :param unfreeze_freq: Frequency of updating the target network.
        :param device: torch.device instance. Either 'torch.device("cuda:n")' where n
        is an integer, (0 in case of single gpu systems) or 'torch.device("cpu")'.
        :param verbose: Agent will log returns after each episode to default logger
        if True. No logging if False.
        :param floating_dtype: Floating point datatype used in the agent. Should match
        with what the value_function expects.
        """
        super(DDQN, self).__init__(tensor_specs={"device": device, "dtype": floating_dtype})
        self.gamma = gamma
        self.epsilon_fn = epsilon_fn

        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.start_training_at = start_training_at
        self.unfreeze_freq = unfreeze_freq

        # 1 step memory.
        self.prev_sa = None

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # Counts the number of actions taken by the agent.
        # Used to calculate epsilon.
        self.acting_steps = 0
        # Used to update target network after every
        # 'unfreeze_step' training steps.
        self.training_steps = 0

        # The network.
        self.action_value_function = value_function
        self.target_network = copy.deepcopy(value_function)

        self.optimizer = optimizer
        self.scheduler = lr_scheduler

        # Set to eval mode by default.
        self.action_value_function.train(False)
        self.target_network.train(False)

        self.ret = 0
        self.verbose = verbose

    def __call__(self, state: np.ndarray, epsilon: float = 0) -> int:
        """
        Call this function during inference.
        :param state: The state of the world, numpy array without the batch dimension.
        :param epsilon: Chance of sub-optimal action selection.
        :return: The index of the action chosen.
        """
        return self.sample_action(state, epsilon)

    def train_step(self, state: np.ndarray, reward: float, episode_ended: bool) -> int:
        """
        Gets called during training, takes action based on world state.
        :param state: The state of the world, numpy array without the batch dimension.
        :param reward: Float scalar reward signal.
        :param episode_ended: True if episode ended, False otherwise.
        :return: Index of action to take.
        """
        if not episode_ended:
            # Sample an action for the current state.
            a = self.sample_action(state, max(0., self.epsilon_fn(self.acting_steps)))
            self.acting_steps += 1
            # Save to replay buffer.
            if self.prev_sa is not None:
                old_state, action = self.prev_sa
                self.replay_buffer.append((old_state, action, reward, state, 1))
            # Remember this for the next round.
            self.prev_sa = (state, a)
            self.replay()
            self.ret += reward
            return a
        if episode_ended:
            # Save to replay buffer.
            if self.prev_sa is not None:
                old_state, action = self.prev_sa
                self.replay_buffer.append((old_state, action, reward, state, 0))

            if self.verbose:
                logging.info(f"Episode ended with return {self.ret}")
            self.ret = 0
            self.prev_sa = None
            self.replay()
            return 0

    def replay(self):
        """
        Performs a single update on a mini batch sampled from the experience replay.
        """
        # If the replay buffer does not contain the minimal amount
        # of experience required for training skip training.
        if len(self.replay_buffer) < self.start_training_at or len(self.replay_buffer) < self.replay_batch_size:
            return
        self.training_steps += 1

        # Update frozen target network if needed.
        if self.training_steps % self.unfreeze_freq == 0:
            self.target_network.load_state_dict(self.action_value_function.state_dict())

        # Sample random batch from replay memory to train on.
        batch = random.sample(self.replay_buffer, self.replay_batch_size)

        # (s_old, action, reward, s_new, non-final indicator) along first axis.
        batch = np.array(batch, dtype=object).T
        s_old = self.tensor(np.stack(batch[0]))
        actions = torch.tensor(np.stack(batch[1]), device=self.tensor_specs["device"], dtype=torch.long)
        reward = self.tensor(np.stack(batch[2]))[..., None]     # One for each entry in batch.
        s_new = self.tensor(np.stack(batch[3]))
        indicator = self.tensor(np.stack(batch[4]))[..., None]     # One for each entry in batch.

        # Creating the target.
        with torch.no_grad():
            next_state_value_idx = self.target_network(s_new)
            next_state_value_idx = torch.max(next_state_value_idx, dim=1)[1]
            next_state_value = self.target_network(s_new)
            next_state_value = next_state_value[range(next_state_value_idx.shape[0]), next_state_value_idx][..., None]

        target = reward + indicator * self.gamma * next_state_value

        # Training pass.
        self.action_value_function.train(True)
        q_values = self.action_value_function(s_old)

        # Now it has shape (B, A) where B is batch size, A is number of actions.
        # Index it with actions taken so we get shape (B, 1), which is what our target
        # shape is. So we only backprop to actions that we actually took.
        q_values = q_values[range(q_values.shape[0]), actions][..., None]

        loss = mse_loss(input=q_values, target=target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.action_value_function.train(False)

    def sample_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Samples an action according to epsilon greedy policy for the
        given state. No gradient is being recorded, don't use it when
        training. Does NOT support batch mode, only samples a single action
        for a single state.
        :param state: The state for which the action needs to be determined.
        No batch dimension.
        :param epsilon: Chance of sub-optimal action selection.
        :return: Index of action to take.
        """
        x = self.tensor(state[None, ...])
        with torch.no_grad():
            y = self.action_value_function(x)
        a = torch.max(y, dim=1)[1].item()
        a = a if random.uniform(0, 1) > epsilon else random.sample(range(len(y)+1), 1)[0]
        return a

    def save(self, path: str) -> None:
        torch.save({
            "q": self.action_value_function.state_dict(),
            "tgt": self.target_network.state_dict(),
            "optim": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "acting_steps": self.acting_steps,
            "training_steps": self.training_steps
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path)
        self.action_value_function.load_state_dict(ckpt["q"])
        self.target_network.load_state_dict(ckpt["tgt"])
        self.optimizer.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.acting_steps = ckpt["acting_steps"]
        self.training_steps = ckpt["training_steps"]
