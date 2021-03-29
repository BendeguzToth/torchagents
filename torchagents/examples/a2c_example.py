"""
Running the A2C agent on cart pole.
"""

# Standard libraries
import logging

# Third-party libraries
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
import gym

# Project files
from a2c import A2C

# Enabling logging.
logging.basicConfig(level=logging.INFO)


def run_a2c(agent: A2C, render: bool = True):
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


def main():
    device = torch.device("cuda:0")
    value_net = Sequential(
        Linear(in_features=4, out_features=24),
        ReLU(),
        Linear(in_features=24, out_features=1)
    ).to(device)

    policy_net = Sequential(
        Linear(in_features=4, out_features=24),
        ReLU(),
        Linear(in_features=24, out_features=2)
    ).to(device)

    optimizer_value = Adam(params=value_net.parameters(), lr=0.005)
    optimizer_policy = Adam(params=policy_net.parameters(), lr=0.001)

    agent = A2C(
        value_net=value_net,
        policy_net=policy_net,
        optimizer_value_net=optimizer_value,
        optimizer_policy_net=optimizer_policy,
        gamma=0.99,
        lambda_value=0,
        lambda_policy=0,
        device=device,
        verbose=True
    )

    run_a2c(agent, render=True)


if __name__ == '__main__':
    main()
