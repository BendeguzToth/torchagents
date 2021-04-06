"""
Advantage Actor-Critic
"""

# Standard libraries
from typing import Optional
import logging

# Third-party dependencies
import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import softmax
from torch.optim import Optimizer

# Project files
from agent import Agent, Scheduler


class A2C(Agent):
    """
    Advantage Actor-Critic with eligibility traces.
    """
    def __init__(
            self,
            value_net: Module,
            policy_net: Module,
            optimizer_value_net: Optimizer,
            optimizer_policy_net: Optimizer,
            gamma: float,
            lambda_value: float,
            lambda_policy: float,
            device: torch.device,
            lr_scheduler_value_net: Optional[Scheduler] = None,
            lr_scheduler_policy_net: Optional[Scheduler] = None,
            verbose: bool = False,
            floating_dtype: torch.dtype = torch.float32
    ):
        """
        Ctor.
        :param value_net: State value function.
        :param policy_net: Policy net. Input of shape (B, F) where B is batch, F is state feature
        vector. Output of shape (B, A) where A is the number of possible actions. Do NOT use softmax
        at the end, output reals.
        :param optimizer_value_net: Value net optimizer.
        :param optimizer_policy_net: Policy net optimizer.
        :param gamma:
        :param lambda_value: TD-style lambda to weigh n-step targets. Uses 1-step target if 0,
        uses mean of all n-step targets if 1.
        :param lambda_policy: TD-style lambda to weigh n-step targets. Uses 1-step target if 0,
        uses mean of all n-step targets if 1.
        :param device: torch.device instance. Either 'torch.device("cuda:n")' where n
        is an integer, (0 in case of single gpu systems) or 'torch.device("cpu")'.
        :param lr_scheduler_value_net: [Optional] Learning rate scheduler.
        :param lr_scheduler_policy_net: [Optional] Learning rate scheduler.
        :param verbose: Agent will log returns after each episode to default logger
        if True. No logging if False.
        :param floating_dtype: Floating point datatype used in the agent. Should match
        with what the value_function expects.
        """
        super(A2C, self).__init__(tensor_specs={"device": device, "dtype": floating_dtype})

        self.value_net = value_net
        self.policy_net = policy_net

        # Networks in eval mode by default.
        self.value_net.train(False)
        self.policy_net.train(False)

        self.optimizer_value = optimizer_value_net
        self.optimizer_policy = optimizer_policy_net
        self.scheduler_value = lr_scheduler_value_net
        self.scheduler_policy = lr_scheduler_policy_net

        self.gamma: float = gamma
        self.lambda_value: float = lambda_value
        self.lambda_policy: float = lambda_policy

        self.eligibility_trace_value = dict()
        for p in self.value_net.parameters():
            self.eligibility_trace_value[p] = torch.zeros_like(p)

        self.eligibility_trace_policy = dict()
        for p in self.policy_net.parameters():
            self.eligibility_trace_policy[p] = torch.zeros_like(p)

        self.device = device
        self.verbose = verbose

        # Caches the previous state and action.
        self.prev_s = None
        self.prev_a = None

        self.ret = 0

    def __call__(self, state, argmax=True) -> int:
        """
        Inference.
        :param state: State numpy array without batch dimension.
        :param argmax: If True argmax will be used to select action
        from probabilities, otherwise it will be sampled from the output
        distribution.
        :return: Action index.
        """
        with torch.no_grad():
            action_distribution = softmax(self.policy_net(self.tensor(state[None, ...])), dim=1)
        if argmax:
            action_idx = torch.max(action_distribution, dim=1)[1].item()
        else:
            action_idx = np.random.choice(np.arange(action_distribution.shape[1]),
                                          p=action_distribution.cpu().numpy()[0])
        return action_idx

    def train_step(self, state: np.ndarray, reward: float, episode_ended: bool) -> int:
        """
        Gets called by the environment, takes action based on world state.
        :param state: The state of the game, numpy array of shape (F,).
        :param reward: Float scalar reward signal.
        :param episode_ended: True when player died, False otherwise.
        :return: Index of action to take.
        """
        self.value_net.train(True)
        self.policy_net.train(True)

        self.ret += reward
        # Add batch dimension to state, convert to tensor.
        state = self.tensor(state[None, ...])
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        if self.prev_a is not None:
            # Propagate previous state through the value network, so that
            # it contains the proper gradients.
            prev_value = self.value_net(self.prev_s)
            # Obtain the value of the current state.
            self.value_net.train(False)
            with torch.no_grad():
                curr_value = self.value_net(state)
            self.value_net.train(True)

            # Create the 1-step TD-error.
            if episode_ended:
                # If episode ended the target is just the last reward. Here added to vector of
                # appropriate size to upcast scalar reward.
                value_target = reward + torch.zeros(size=(1, 1), dtype=self.tensor_specs['dtype'], device=self.tensor_specs['device'])
            else:
                # Else the target is just 1-step TD error.
                value_target = reward + self.gamma * curr_value
            # Backprop here to obtain ∇w [v(s)].
            prev_value.backward()
            # E_t = Gamma*Lambda*E_t-1 + ∇w [v(s)].
            self.update_eligibility_trace_value()
            td_error = (value_target - prev_value).detach()[0]
            self.apply_eligibility_trace_value(td_error)
            self.optimizer_value.step()
            if self.scheduler_value is not None:
                self.scheduler_value.step()

            # Policy update.

            # Negative because gradient ascent. It is multiplied with the advantage in the function
            # apply_eligibility_trace_policy(td_error).
            loss_policy = -torch.log(softmax(self.policy_net(self.prev_s), dim=1)[0, self.prev_a])
            loss_policy.backward()
            self.update_eligibility_trace_policy()
            self.apply_eligibility_trace_policy(td_error)
            self.optimizer_policy.step()
            if self.scheduler_policy is not None:
                self.scheduler_policy.step()

        # Here just take an action.
        self.value_net.train(False)
        self.policy_net.train(False)
        with torch.no_grad():
            action_distribution = softmax(self.policy_net(state), dim=1)
        self.policy_net.train(True)
        action_idx = np.random.choice(np.arange(action_distribution.shape[1]), p=action_distribution.detach().cpu().numpy()[0])

        self.prev_s = state
        self.prev_a = action_idx

        if episode_ended:
            if self.verbose:
                logging.info(f"Episode ended with return {self.ret}")
            self.ret = 0
            self.prev_s = None
            self.prev_a = None
            # Reset eligibility trace.
            for p in self.value_net.parameters():
                self.eligibility_trace_value[p] = torch.zeros_like(p)
            for p in self.policy_net.parameters():
                self.eligibility_trace_policy[p] = torch.zeros_like(p)
        return action_idx

    def update_eligibility_trace_value(self):
        """
        Updates each value of the eligibility trace of the value net from the
        current gradient of the parameters.
        """
        for k in self.eligibility_trace_value.keys():
            self.eligibility_trace_value[k] *= self.gamma * self.lambda_value
            self.eligibility_trace_value[k] += k.grad

    def apply_eligibility_trace_value(self, td_error):
        """
        Overrides gradients for optimization.
        """
        with torch.no_grad():
            for k in self.eligibility_trace_value.keys():
                k.grad *= 0
                k.grad -= self.eligibility_trace_value[k] * td_error

    def update_eligibility_trace_policy(self):
        """
        Updates the policy trace from the current grad which should be
        ∇θ [ln policy(prev_s, prev_a)]
        """
        for k in self.eligibility_trace_policy.keys():
            self.eligibility_trace_policy[k] *= self.lambda_policy
            self.eligibility_trace_policy[k] += k.grad

    def apply_eligibility_trace_policy(self, td_error):
        """
        Overrides gradients for optimization.
        """
        with torch.no_grad():
            for k in self.eligibility_trace_policy.keys():
                k.grad *= 0
                k.grad += self.eligibility_trace_policy[k] * td_error

    def save(self, path: str) -> None:
        torch.save({
            "value_net": self.value_net,
            "policy_net": self.policy_net,
            "optimizer_value": self.optimizer_value.state_dict(),
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "scheduler_value": self.scheduler_value.state_dict() if self.scheduler_value is not None else None,
            "scheduler_policy": self.scheduler_value.state_dict() if self.scheduler_policy is not None else None
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path)
        self.value_net.load_state_dict(ckpt["value_net"])
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.optimizer_value.load_state_dict(ckpt["optimizer_value"])
        self.optimizer_policy.load_state_dict(ckpt["optimizer_policy"])
        if self.scheduler_value is not None:
            self.scheduler_value.load_state_dict(ckpt["scheduler_value"])
        if self.scheduler_policy is not None:
            self.scheduler_value.load_state_dict(ckpt["scheduler_policy"])
