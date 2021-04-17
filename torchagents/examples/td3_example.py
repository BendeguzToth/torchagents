"""
Running TD3 on pendulum.
"""

# Standard libraries
import logging

# Third-party libraries
import torch
from torch.nn import Linear, ReLU, Sequential
from torch.optim import Adam
import gym

# Project files
from td3 import TD3

# Enabling logging.
logging.basicConfig(level=logging.INFO)


def run_td3(agent: TD3, render: bool = True):
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
    value_function_1 = Sequential(
        Linear(in_features=4, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=128),
        ReLU(),
        Linear(in_features=128, out_features=1)
        ).to(torch.device("cuda:0"))

    value_function_2 = Sequential(
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

    optimizer_value_1 = Adam(params=value_function_1.parameters(), lr=0.0003)
    optimizer_value_2 = Adam(params=value_function_2.parameters(), lr=0.0003)
    optimizer_policy = Adam(params=policy_function.parameters(), lr=0.0003)

    agent = TD3(
        value_net_1=value_function_1,
        value_net_2=value_function_2,
        policy_net=policy_function,
        optimizer_value_net_1=optimizer_value_1,
        optimizer_value_net_2=optimizer_value_2,
        optimizer_policy_net=optimizer_policy,
        lr_scheduler_value_net_1=None,
        lr_scheduler_value_net_2=None,
        lr_scheduler_policy_net=None,
        gamma=0.99,
        noise_std_f=lambda x: 0.1,
        target_policy_smoothing_std=0.2,
        target_policy_smoothing_bound=0.5,
        policy_update_frequency=2,
        tau=0.005,
        min_action=-2,
        max_action=2,
        replay_buffer_size=10000,
        replay_batch_size=64,
        start_training_at=1000,
        device=torch.device("cuda:0"),
        verbose=True
    )

    run_td3(agent, render=True)


if __name__ == '__main__':
    main()
