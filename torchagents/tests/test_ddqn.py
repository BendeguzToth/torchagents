"""
Testing the Double DQN agent on cart pole.
"""

# Standard libraries
import logging

# Third-party libraries
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
import gym

# Project files
from ddqn import DDQN

# Enabling logging.
logging.basicConfig(level=logging.INFO)


def test_ddqn(agent: DDQN, render: bool = True):
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
    value_function = Sequential(
            Linear(in_features=4, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=32),
            ReLU(),
            Linear(in_features=32, out_features=2)
        ).to(torch.device("cuda:0"))

    optimizer = RMSprop(params=value_function.parameters(), alpha=0.95, lr=0.0001)

    agent = DDQN(
        value_function=value_function,
        optimizer=optimizer,
        lr_scheduler=LambdaLR(optimizer=optimizer, lr_lambda=lambda e: max(0.9999 ** e, 0.1)),
        gamma=0.95,
        epsilon_fn=lambda x: 0.9999**x,
        replay_buffer_size=10000,
        replay_batch_size=128,
        start_training_at=1024,
        unfreeze_freq=64,
        device=torch.device("cuda:0"),
        verbose=True
    )

    test_ddqn(agent, render=True)


if __name__ == '__main__':
    main()
