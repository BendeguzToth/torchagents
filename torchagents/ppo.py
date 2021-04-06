"""
PPO-Clip implementation with truncated GAE.
"""
# Standard libraries
from collections import deque
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


class PPO(Agent):
    """
    PPO-Clip algorithm.
    Uses GAE truncated up to end of episode or end of batch.
    Critic trains towards lambda weighted TD-targets up to episode end
    or end of batch. Lambda is shared between the two.
    The network argument should be a single model with a 2-tuple (state_value, policy)
    as output.
    """
    def __init__(
            self,
            net: Module,
            optimizer: Optimizer,
            c1: float,
            c2: float,
            gamma: float,
            lambda_: float,
            epsilon: float,
            run_for_t: int,
            train_for_n_epochs: int,
            batch_size: int,
            device: torch.device,
            scheduler: Optional[Scheduler] = None,
            verbose: bool = False,
            floating_dtype: torch.dtype = torch.float32
    ):
        """
        Ctor.
        :param net: Single network object that takes a stake stack as argument,
        and produces a tuple of (value, policy) outputs.
        :param optimizer: The optimizer to use during training.
        :param c1: Multiplier for the gradient from the critic head during
        training. (Relative to actor, that is fixed at 1.)
        :param c2: Coefficient of the entropy regularizer.
        :param gamma: Per step reward discount.
        :param lambda_: Lambda for weighting n-step returns for both the critic head
        and the advantage for the actor.
        :param epsilon: Clipping value.
        :param run_for_t: Model trains after every t steps on the WHOLE buffer.
        :param train_for_n_epochs: Trains for n epochs every training loop.
        :param batch_size: The batch size used during training.
        representation. Should be same as temporal dim of network.
        :param device: Device of the network. Either torch.device('cpu') or
        torch.device('cuda:n') where n is a number, usually 0 in case of single gpu.
        :param scheduler: Optional scheduler parameter.
        :param verbose: If True it will log undiscounted returns at the end of episode.
        Will not log if False.
        :param floating_dtype: Default datatype to use for floating point numbers.
        Defaults to float32.
        """
        super(PPO, self).__init__({"device": device, "dtype": floating_dtype})
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2

        self.train_for_n_epochs = train_for_n_epochs
        self.batch_size = batch_size

        # (s, a, r, s', πθ(s, a), episode_ended)
        self.memory = deque(maxlen=run_for_t)

        # Undiscounted cumulative reward.
        self.ret = 0

        # Log previous values to create 1-step target.
        self.prev_s = None
        self.prev_a = None
        self.prev_prob = None

        self.device = device
        self.verbose = verbose

        self.net.to(self.device)

    def __call__(self, state) -> int:
        """
        Call this for inference.
        :param state: Numpy array of shape (F,) where F is the number of
        features in the state representation.
        :return: Action index.
        """
        self.net.train(False)
        with torch.no_grad():
            _, policy = self.net(torch.tensor(state[None, ...], dtype=torch.float32, device=self.device))
        action_distribution = torch.softmax(policy, dim=1)
        action_idx = np.random.choice(np.arange(action_distribution.shape[1]), p=action_distribution.cpu().numpy()[0])
        return action_idx

    def train_step(self, state: np.ndarray, reward: float, episode_ended: bool) -> int:
        """
        Gets called by the environment, takes action based on world state.
        :param state: The state of the game, numpy array of shape (F,).
        :param reward: Float scalar reward signal.
        :param episode_ended: True when player died, False otherwise.
        :return: Index of the chosen action.
        """
        self.ret += reward
        if not episode_ended:
            self.net.train(False)
            with torch.no_grad():
                _, policy = self.net(torch.tensor(state[None, ...], dtype=torch.float32, device=self.device))
            action_distribution = torch.softmax(policy, dim=1)
            action_idx = np.random.choice(np.arange(action_distribution.shape[1]), p=action_distribution.cpu().numpy()[0])

            # Add to memory.
            if self.prev_s is not None:
                self.memory.append((self.prev_s, self.prev_a, reward, state, self.prev_prob.detach().cpu().numpy(), False))

            self.prev_s = state
            self.prev_a = action_idx
            self.prev_prob = action_distribution[0, action_idx]

            if len(self.memory) == self.memory.maxlen:
                self._train()
            return action_idx

        if episode_ended:
            self.memory.append((self.prev_s, self.prev_a, reward, state, self.prev_prob.detach().cpu(), True))
            self.prev_s = None
            self.prev_a = None
            self.prev_prob = None
            if self.verbose:
                logging.info(f"Episode ended with return {self.ret}")
            self.ret = 0

            if len(self.memory) == self.memory.maxlen:
                self._train()
            return 0

    def _train(self):
        # First obtain the TD-error.
        x = np.array(self.memory, dtype=object)
        reward = np.stack(x[:, 2])[..., None]   # (run_for_t, 1)
        s_new = np.stack(x[:, 3])               # (run_for_t, 4, 64, 64)
        ended = np.stack(x[:, 5])[..., None]    # (run_for_t, 1)

        # Pre-compute the TD-targets with the old network. Train towards these
        # during the entire training session.
        self.net.train(False)
        with torch.no_grad():
            new_state_value, _ = self.net(torch.tensor(s_new, dtype=torch.float32, device=self.device))
        td_target = torch.where(torch.tensor(ended, device=self.device), torch.tensor(reward, dtype=torch.float32, device=self.device), torch.tensor(reward, dtype=torch.float32, device=self.device) + self.gamma * new_state_value)   # (run_for_t, 1)

        # EPOCH
        for e in range(self.train_for_n_epochs):
            # Create batches.
            batch_start_indices = np.arange(0, len(self.memory), self.batch_size)
            np.random.shuffle(batch_start_indices)
            # MINI BATCH
            for start in range(len(batch_start_indices)):
                # Get the data that is part of the current mini batch.
                s_old = np.stack(x[start:start+self.batch_size, 0])             # (run_for_t, 4)
                a_old = np.stack(x[start:start+self.batch_size, 1])[..., None]  # (run_for_t, 1)
                prob = np.stack(x[start:start+self.batch_size, 4])[..., None]   # (run_for_t, 1)
                ended = np.stack(x[start:start+self.batch_size, 5])[..., None]  # (run_for_t, 1)
                td_targets_batch = td_target[start:start+self.batch_size]       # (run_for_t, 1)

                # Update the value function, obtain estimates for the old states.
                self.net.train(True)
                old_state_value, policy = self.net(torch.tensor(s_old, dtype=torch.float32, device=self.device))
                current_policy = softmax(policy, dim=1)

                td_error = td_targets_batch - old_state_value  # (run_for_t, 1)

                # Now we chop up the data into chunks that end when an episode ended.
                # It is needed to know when to stop propagating the advantage.
                ends = np.where(ended == True)[0] + 1
                ends = ends.tolist()
                ends.append(ended.shape[0])
                a = list(set(ends))
                a.sort()
                b = [0] + a[:-1]
                lengths = np.array(a, dtype=int) - np.array(b, dtype=int)
                error_chunks = torch.split(td_error, split_size_or_sections=lengths.tolist())
                error_matrices = map(lambda x: torch.repeat_interleave(x.T, repeats=x.shape[0], dim=0), filter(lambda y: y.shape[0] > 0, error_chunks))

                # error_matrices is now a list of matrices, each of them ending when the episode ends. Each column contains the one-step
                # TD-error of the corresponding entry, and there are as many rows to make it a square matrix. Example:
                # |𝛿1 𝛿2 𝛿3 ... |
                # |𝛿1 𝛿2 𝛿3 ... |
                #  .  .  .      |
                #  .  .  .      |
                #  .  .  .      |
                # Where 𝛿n is the 1 step TD-error of the n-th sample in the episode. (It starts from one in every matrix in the list,
                # and goes up to its size, which is variable.)
                # Now we zero out the lower triangle to obtain:
                # | 𝛿1 𝛿2 𝛿3
                # |  0 𝛿2 𝛿3
                # |  0  0 𝛿3
                # Each row now contains the relevant error terms in the advantage. For the first time step we need all 3 (𝛿1, 𝛿2, 𝛿3)
                # the second time step only depends on errors after it (𝛿2, 𝛿3) etc. Then we can multiply it by a matrix containing the
                # γλ terms, and we get the advantage.
                def make_triu(m):
                    if type(m) is torch.Tensor:
                        t = torch
                    else:
                        t = np
                    a = t.zeros_like(m)
                    a[np.triu_indices(m.shape[0])] = 1
                    return m * a

                error_matrices = list(map(make_triu, error_matrices))
                # Squaring for the critic loss.
                critic_loss = list(map(lambda x: torch.pow(x, 2), error_matrices))
                # Detach so that the advantage calculations do not backprop into the critic.
                error_matrices = list(map(lambda x: x.detach(), error_matrices))

                # Here we calculate the coefficient of the γλ term. The goal is to obtain a matrix of the form
                # 1 2 3 4
                # . 1 2 3
                # . . 1 2
                # The dots can be anything as they will get multiplied by zero.
                powers = map(lambda x: np.repeat(np.arange(x.shape[0])[None, ...], repeats=x.shape[0], axis=0), error_matrices)
                # Now it is
                # 0 1 2 3 4 ...
                # 0 1 2 3 4 ...
                # 0 1 2 3 4 ...

                # So subtract
                # 0 0 0 0 0 ...
                # 1 1 1 1 1 ...
                # 2 2 2 2 2 ...
                powers = map(lambda x: x - np.repeat(np.arange(x.shape[0])[..., None], repeats=x.shape[0], axis=1), powers)
                powers = list(map(make_triu, powers))
                error_coefficients = map(lambda x: torch.tensor(np.power(self.gamma * self.lambda_, x), dtype=torch.float32, device=self.device), powers)
                error_coefficients = list(map(make_triu, error_coefficients))

                adv_matrices = list(map(lambda x: x[0] * x[1], zip(error_coefficients, error_matrices)))
                critic_loss = list(map(lambda x: x[0] * x[1], zip(error_coefficients, critic_loss)))

                advantages = [torch.sum(x, dim=1, keepdim=False) for x in adv_matrices]
                advantage = torch.cat(advantages)[..., None].detach()  # (run_for_t, 1)

                critic_loss = [torch.sum(x, dim=1, keepdim=False) for x in critic_loss]  # (run_for_t, 1)
                critic_loss = torch.cat(critic_loss).mean()

                # The policy net training.
                imp_a = current_policy[np.array(range(prob.shape[0]))[..., None], a_old]
                imp_b = torch.tensor(prob, dtype=torch.float32, device=self.device)
                importance_sampling_ratio = imp_a / imp_b
                surr1 = importance_sampling_ratio * advantage
                surr2 = torch.clamp(importance_sampling_ratio, 1-self.epsilon, 1+self.epsilon) * advantage
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                loss = actor_loss + self.c1 * critic_loss + self.c2 * torch.mean(torch.sum(torch.log(current_policy) * current_policy, dim=1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.net.train(False)

        # Throw away old memory.
        self.memory.clear()

    def save(self, path: str) -> None:
        torch.save({
            "net": self.net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optim"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
