"""
Implementation of Twin Delayed Deep Deterministic Policy Gradient.
"""

# Standard libraries
from typing import Optional, Callable
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
from agent import Agent, Action, Scheduler


class TD3(Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient.
    """
    def __init__(
            self,
            value_net_1:      Module,
            value_net_2:      Module,
            policy_net:     Module,
            optimizer_value_net_1:    Optimizer,
            optimizer_value_net_2:    Optimizer,
            optimizer_policy_net:   Optimizer,
            lr_scheduler_value_net_1: Optional[Scheduler],
            lr_scheduler_value_net_2: Optional[Scheduler],
            lr_scheduler_policy_net: Optional[Scheduler],
            gamma:          float,
            tau:         float,
            noise_std_f:    Callable[[int], float],
            target_policy_smoothing_std:    float,
            target_policy_smoothing_bound:  float,
            policy_update_frequency:        int,
            min_action:     float,
            max_action:     float,
            replay_buffer_size: int,
            replay_batch_size:  int,
            start_training_at:  int,
            device:             torch.device,
            verbose:            bool = False,
            floating_dtype:     torch.dtype = torch.float32
    ):
        """
        Ctor.
        :param value_net_1: Torch module, takes input of shape (B, S+A) where
        B is the batch dimensions, S is the size of the state vector and A is the
        size of the action vector. Returns a tensor of shape (B, 1), the value
        of the action in the state. Deterministic policy gradient calculated with
        respect to this value function. Trained in conjunction with 'value_net_2'
        double q-learning style.
        :param value_net_2: Torch module, takes input of shape (B, S+A) where
        B is the batch dimensions, S is the size of the state vector and A is the
        size of the action vector. Returns a tensor of shape (B, 1), the value
        of the action in the state. Trained in conjunction with 'value_net_1'
        double q-learning style.
        :param policy_net: Torch module, takes input of shape (B, S) where
        B is the batch dimensions, S is the size of the state vector. Returns
        a tensor of (B, A) where A is the number of actions.
        :param optimizer_value_net_1: Optimizer to use for training value net 1.
        :param optimizer_value_net_2: Optimizer to use for training value net 2.
        :param optimizer_policy_net: Optimizer to use for training the policy net.
        :param lr_scheduler_value_net_1: [Optional] Learning rate scheduler. Can be None.
        :param lr_scheduler_value_net_2: [Optional] Learning rate scheduler. Can be None.
        :param lr_scheduler_policy_net: [Optional] Learning rate scheduler. Can be None.
        :param gamma: Reward discount rate. [0, 1]
        :param tau: Weights for smooth target update. It is the ratio of new target
        weights (should be close to 0).
        :param noise_std_f: A function that returns the standard deviation of the added noise
        during training based on the training step as parameter.
        :param target_policy_smoothing_std: Standard deviation of the noise added to the action
        during creating target for value networks.
        :param target_policy_smoothing_bound: Target policy noise will be clipped to have smaller
        magnitude than this value.
        :param policy_update_frequency: Policy network update frequency wrt value update.
        All target networks update together with he policy net.
        :param min_action: Lower bound of valid action range. Output of the policy net will
        be clipped to have at least this value.
        :param max_action: Upper bound of valid action range. Output of the policy net will
        be clipped to have at most this value.
        :param replay_buffer_size: Size of the replay buffer.
        :param replay_batch_size: Batch size used when training from the replay
        buffer.
        :param start_training_at: Start training after at least this amount of data
        in experience buffer.
        :param device: torch.device instance. Either 'torch.device("cuda:n")' where n
        is an integer, (0 in case of single gpu systems) or 'torch.device("cpu")'.
        :param verbose: Agent will log returns after each episode to default logger
        if True. No logging if False.
        :param floating_dtype: Floating point datatype used in the agent. Should match
        with what the value_function expects.
        """
        super(TD3, self).__init__(tensor_specs={"device": device, "dtype": floating_dtype})
        self.gamma = gamma
        self.tau = tau
        self.noise_std_f = noise_std_f
        self.target_policy_smoothing_std = target_policy_smoothing_std
        self.target_policy_smoothing_bound = target_policy_smoothing_bound
        self.policy_update_frequency = policy_update_frequency
        self.min_action = min_action
        self.max_action = max_action

        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.start_training_at = start_training_at

        # 1 step memory.
        self.prev_sa = None

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        # Counts the number of actions taken by the agent.
        # Used to calculate noise std.
        self.acting_steps = 0
        self.update_steps = 0

        # The networks.
        self.value_net_1 = value_net_1
        self.value_net_2 = value_net_2
        self.policy_net = policy_net
        self.target_value_net_1 = copy.deepcopy(value_net_1)
        self.target_value_net_2 = copy.deepcopy(value_net_2)
        self.target_policy_net = copy.deepcopy(policy_net)

        self.optimizer_value_net_1 = optimizer_value_net_1
        self.optimizer_value_net_2 = optimizer_value_net_2
        self.optimizer_policy_net = optimizer_policy_net
        self.scheduler_value_net_1 = lr_scheduler_value_net_1
        self.scheduler_value_net_2 = lr_scheduler_value_net_2
        self.scheduler_policy_net = lr_scheduler_policy_net

        # Set to eval mode by default.
        self.value_net_1.train(False)
        self.value_net_2.train(False)
        self.policy_net.train(False)
        self.target_value_net_1.train(False)
        self.target_value_net_2.train(False)
        self.target_policy_net.train(False)

        self.ret = 0
        self.verbose = verbose

    def __call__(self, state: np.ndarray, epsilon: float = 0) -> Action:
        """
        Call this function during inference.
        :param state: The state of the world, numpy array without the batch dimension. (F,)
        :param epsilon: Standard deviation of added noise.
        :return: The action.
        """
        return self.sample_action(state, epsilon)

    def train_step(self, state: np.ndarray, reward: float, episode_ended: bool) -> Action:
        """
        Gets called during training, takes action based on world state.
        :param state: The state of the world, numpy array without the batch dimension. (F,)
        :param reward: Float scalar reward signal.
        :param episode_ended: Indicates whether this is the last state in the episode.
        :return: Action to take.
        """
        a = self.sample_action(state, max(0., self.noise_std_f(self.acting_steps)))
        if not episode_ended:
            self.acting_steps += 1
            # Save to replay buffer.
            if self.prev_sa is not None:
                old_state, action = self.prev_sa
                self.replay_buffer.append((old_state, action, reward, state, 1))
            # Remember this for the next round.
            self.prev_sa = (state, a)
            self.replay()
            self.ret += reward
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
        return a

    def replay(self):
        # If the replay buffer does not contain the minimal amount
        # of experience required for training skip training.
        if len(self.replay_buffer) < self.start_training_at or len(self.replay_buffer) < self.replay_batch_size:
            return

        self.update_steps += 1

        # Sample random batch from replay memory to train on.
        batch = random.sample(self.replay_buffer, self.replay_batch_size)

        # (s_old, action, reward, s_new, non-final indicator) along first axis.
        batch = np.array(batch, dtype=object).T
        s_old = self.tensor(np.stack(batch[0]))
        actions = self.tensor(np.stack(batch[1]))
        reward = self.tensor(np.stack(batch[2]))[..., None]         # One for each entry in batch.
        s_new = self.tensor(np.stack(batch[3]))
        indicator = self.tensor(np.stack(batch[4]))[..., None]      # One for each entry in batch.

        # Creating the target.
        with torch.no_grad():
            next_action = self.target_policy_net(s_new)
            noise = torch.clamp(torch.randn_like(next_action) * self.target_policy_smoothing_std, -self.target_policy_smoothing_bound, self.target_policy_smoothing_bound)
            next_action = torch.clamp(next_action + noise, self.min_action, self.max_action)
            value_1 = self.target_value_net_1(torch.cat([s_new, next_action], dim=1))
            value_2 = self.target_value_net_2(torch.cat([s_new, next_action], dim=1))
            next_state_value = torch.minimum(value_1, value_2)

        target = reward + indicator * self.gamma * next_state_value     # (B, 1)

        # Training critics.
        self.value_net_1.train(True)
        self.value_net_1.train(True)

        estimates_1 = self.value_net_1(torch.cat([s_old, actions], dim=1))
        estimates_2 = self.value_net_2(torch.cat([s_old, actions], dim=1))

        loss_value_1 = mse_loss(input=estimates_1, target=target)
        loss_value_2 = mse_loss(input=estimates_2, target=target)

        self.optimizer_value_net_1.zero_grad()
        self.optimizer_value_net_2.zero_grad()

        loss_value_1.backward()
        loss_value_2.backward()

        self.optimizer_value_net_1.step()
        if self.scheduler_value_net_1:
            self.scheduler_value_net_1.step()

        self.optimizer_value_net_2.step()
        if self.scheduler_value_net_2:
            self.scheduler_value_net_2.step()

        self.value_net_1.train(False)
        self.value_net_2.train(False)

        # Training actor and update targets.
        if self.update_steps % self.policy_update_frequency == 0:
            self.policy_net.train(True)

            current_actions = torch.clamp(self.policy_net(s_old), self.min_action, self.max_action)
            loss_policy = -torch.mean(self.value_net_1(torch.cat([s_old, current_actions], dim=1)))

            self.optimizer_policy_net.zero_grad()
            loss_policy.backward()
            self.optimizer_policy_net.step()
            if self.scheduler_policy_net:
                self.scheduler_policy_net.step()

            self.policy_net.train(False)

            # Update the target networks.
            with torch.no_grad():
                for p, q in zip(self.target_value_net_1.parameters(), self.value_net_1.parameters()):
                    p *= 1 - self.tau
                    p += self.tau * q

                for p, q in zip(self.target_value_net_2.parameters(), self.value_net_2.parameters()):
                    p *= 1 - self.tau
                    p += self.tau * q

                for p, q in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                    p *= 1 - self.tau
                    p += self.tau * q

    def sample_action(self, state: np.ndarray, epsilon: float) -> Action:
        """
        Takes action according to the policy net.
        No gradient is being recorded, don't use it when training.
        :param state: The state for which the action needs to be determined.
        No batch dimension.
        :param epsilon: Noise variance.
        :return: A flat array of action(s).
        """
        x = self.tensor(state[None, ...])
        with torch.no_grad():
            y = self.policy_net(x)
        y = torch.clamp(y, self.min_action, self.max_action)
        noise = torch.randn_like(y) * epsilon
        y = y + noise
        return torch.clamp(y, self.min_action, self.max_action).cpu().numpy().flatten()

    def save(self, path: str) -> None:
        torch.save({
            "value_net_1": self.value_net_1.state_dict(),
            "value_net_2": self.value_net_2.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "target_value_net_1": self.target_value_net_1.state_dict(),
            "target_value_net_2": self.target_value_net_2.state_dict(),
            "target_policy_net": self.target_policy_net.state_dict(),
            "optimizer_value_net_1": self.optimizer_value_net_1.state_dict(),
            "optimizer_value_net_2": self.optimizer_value_net_2.state_dict(),
            "optimizer_policy_net": self.optimizer_policy_net.state_dict(),
            "scheduler_value_net_1": self.scheduler_value_net_1.state_dict() if self.scheduler_value_net_1 is not None else None,
            "scheduler_value_net_2": self.scheduler_value_net_2.state_dict() if self.scheduler_value_net_2 is not None else None,
            "scheduler_policy_net": self.scheduler_policy_net.state_dict() if self.scheduler_policy_net is not None else None,
            "steps": self.acting_steps
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path)
        self.value_net_1.load_state_dict(ckpt["value_net_1"])
        self.value_net_2.load_state_dict(ckpt["value_net_2"])
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_value_net_1.load_state_dict(ckpt["target_value_net_1"])
        self.target_value_net_2.load_state_dict(ckpt["target_value_net_2"])
        self.target_policy_net.load_state_dict(ckpt["target_policy_net"])
        self.optimizer_value_net_1.load_state_dict(ckpt["optimizer_value_net_1"])
        self.optimizer_value_net_2.load_state_dict(ckpt["optimizer_value_net_2"])
        self.optimizer_policy_net.load_state_dict(ckpt["optimizer_policy_net"])
        if self.scheduler_value_net_1 is not None:
            self.scheduler_value_net_1.load_state_dict(ckpt["scheduler_value_net_1"])
        if self.scheduler_value_net_2 is not None:
            self.scheduler_value_net_2.load_state_dict(ckpt["scheduler_value_net_2"])
        if self.scheduler_policy_net is not None:
            self.scheduler_policy_net.load_state_dict(ckpt["scheduler_policy_net"])
        self.acting_steps = ckpt["steps"]
