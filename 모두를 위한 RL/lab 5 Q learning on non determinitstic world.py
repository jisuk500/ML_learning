# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:37:21 2020

@author: jisuk
"""

# %% modules setup

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# register(
#     id='FrozenLake-v3',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4',
#             'is_slippery': True}
# )
env = gym.make('FrozenLake-v0')


# %% initalize table with zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# set learning parameters
num_episodes = 2000

# discount factor
dis = 0.99

# learning rate
learning_rate = 0.85

# %%create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # the Q-table learning algorithm
    while not done:
        # choose and action by greedily (with noise) picking from Q table
        action = np.argmax(
            Q[state, :] + np.random.randn(1, env.action_space.n)/(i+1))

        # get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # update Q-table with new knowledge using learning rate
        Q[state, action] = (1-learning_rate)*Q[state, action] \
            + learning_rate*(reward + dis*np.max(Q[new_state, :]))

        rAll += reward
        state = new_state

    rList.append(rAll)


# %% result  reporting

print("Success Rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

real_viz = Q.reshape([4, 4, -1])
real_viz_ = np.zeros(shape=(12, 12))

for i in range(4):
    for j in range(4):
        real_viz_[1+3*i, 1+3*j - 1] = real_viz[i, j, 0]
        real_viz_[1+3*i + 1, 1+3*j] = real_viz[i, j, 1]
        real_viz_[1+3*i, 1+3*j + 1] = real_viz[i, j, 2]
        real_viz_[1+3*i - 1, 1+3*j] = real_viz[i, j, 3]
        real_viz_[1+3*i, 1+3*j] = (4*i + j + 1) / 100.

env.render()

plt.bar(range(len(rList)), rList, color="blue")
plt.show()

# %%

real_viz = Q.reshape([4, 4, -1])
real_viz_ = np.zeros(shape=(12, 12))

for i in range(4):
    for j in range(4):
        real_viz_[1+3*i, 1+3*j - 1] = real_viz[i, j, 0]
        real_viz_[1+3*i + 1, 1+3*j] = real_viz[i, j, 1]
        real_viz_[1+3*i, 1+3*j + 1] = real_viz[i, j, 2]
        real_viz_[1+3*i - 1, 1+3*j] = real_viz[i, j, 3]
        real_viz_[1+3*i, 1+3*j] = (4*i + j + 1) / 100.

env.render()
