"""
Running PPO on cart pole.
"""

# Standard libraries
import logging

# Third-party libraries
import torch
from torch.nn import Linear, ReLU, Sequential, Module
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import gym

# Project files
from ppo import PPO

# Enabling logging.
logging.basicConfig(level=logging.INFO)


def run_ppo(agent: PPO, render: bool = True):
    env = gym.make("CartPole-v1")
    draw = env.render if render else lambda: ...

    # Train forever.
    while True:
        next_state = env.reset()
        reward = 0
        done = False
        while True:
            action = agent.train_step(state=next_state, reward=reward, episode_ended=done)
            if done:
                break
            next_state, reward, done, info = env.step(action)
            draw()


class PPONet(Module):
    """
    Sequential policy and value net.
    PPO expects a single network with 2 outputs.
    """
    def __init__(self, device):
        super(PPONet, self).__init__()
        self.value_net = Sequential(
            Linear(in_features=4, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=1)
        ).to(device)

        self.policy_net = Sequential(
            Linear(in_features=4, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=2)
        ).to(device)

    def forward(self, x):
        return self.value_net(x), self.policy_net(x)


def main():
    net = PPONet(torch.device("cuda:0"))

    optimizer = Adam(params=net.parameters(), lr=1e-4)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda e: max(0.9999**e, 0.1))

    agent = PPO(
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        c1=0.5,
        c2=0,
        gamma=1,
        lambda_=0.99,
        epsilon=0.05,
        run_for_t=16,
        train_for_n_epochs=4,
        batch_size=16,
        verbose=True,
        device=torch.device("cuda:0")
    )

    run_ppo(agent, render=True)


if __name__ == '__main__':
    main()
