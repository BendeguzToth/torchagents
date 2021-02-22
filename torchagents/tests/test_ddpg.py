"""
Testing the Deep Q-Learning agent on cart pole.
"""

# Standard libraries
import logging

# Third-party libraries
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
import gym

# Project files
from ddpg import DDPG

# Enabling logging.
logging.basicConfig(level=logging.INFO)


def test_ddpg(agent: DDPG, render: bool = True):
    env = gym.make("Pendulum-v0")
    draw = env.render if render else lambda: ...

    # Train forever.
    while True:
        next_state = env.reset()
        reward = 0
        done = False
        ret = 0
        while True:
            action = agent.train_step(state=next_state.flatten(), reward=reward, episode_ended=done)
            if done:
                break
            next_state, reward, done, info = env.step(action)
            ret += reward
            draw()


def main():
    value_function = Sequential(
        Linear(in_features=4, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=1)
        ).to(torch.device("cuda:0"))

    policy_function = Sequential(
        Linear(in_features=3, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=1)
    ).to(torch.device("cuda:0"))

    optimizer_value = Adam(params=value_function.parameters(), lr=0.0003)
    optimizer_policy = Adam(params=policy_function.parameters(), lr=0.0003)

    agent = DDPG(
        value_net=value_function,
        policy_net=policy_function,
        optimizer_value_net=optimizer_value,
        optimizer_policy_net=optimizer_policy,
        lr_scheduler_value_net=None,
        lr_scheduler_policy_net=None,
        gamma=0.99,
        noise_std_f=lambda x: 0.1,
        polyak=0.995,
        min_action=-2,
        max_action=2,
        replay_buffer_size=10000,
        replay_batch_size=64,
        start_training_at=1000,
        clip_policy_grad=0.5,
        clip_value_grad=1,
        device=torch.device("cuda:0"),
        verbose=True
    )

    test_ddpg(agent, render=True)


if __name__ == '__main__':
    main()
