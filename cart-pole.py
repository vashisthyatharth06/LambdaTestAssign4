import gym

import numpy as np
import os

# Common imports
import numpy as np
import random
import os
import collections

env = gym.make('CartPole-v0')
env.reset()
N_scenario = 1000
MAX_ACTIONS = 500
def test_policy(policy_func, n_scenario = N_scenario, max_actions = MAX_ACTIONS, verbose=False):
    final_rewards = []
    for episode in range(n_scenario):
        if verbose and episode % 50 == 0:
            print(episode)
        episode_rewards = 0
        obs = env.reset()  # reset to a random position
        for step in range(max_actions):
            action = policy_func(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        final_rewards.append(episode_rewards)
    return final_rewards

def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1


env.seed(42)
random.seed(0)

# the cart-pole experiment will end if it lasts more than 500 steps, with info="'TimeLimit.truncated': True"
theta_omega_rewards = test_policy(theta_omega_policy, max_actions=510)
print("Average Reward:", sum(theta_omega_rewards) / len(theta_omega_rewards))

env.close()
