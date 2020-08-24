# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:56:33 2020

@author: jisuk
"""

# %% modules import

import tensorflow as tf
from tensorflow.keras import Model, layers

import matplotlib.pyplot as plt
import numpy as np

import random as pr

import gym
env = gym.make('CartPole-v0')

# %% constants defining for policy network
learning_rate = 0.1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

n_hidden_1 = 256
n_hidden_2 = 256

# %% constants defining for Q-learning

num_episodes = 2000
dis = 0.9
rList = []



# %% prepare Q network

class NeuralNet(Model):
    # set layers
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = layers.Dense(n_hidden_1,activation=tf.nn.sigmoid,use_bias=True)
        self.fc2 = layers.Dense(n_hidden_2,activation=tf.nn.sigmoid,use_bias=True)
        self.out = layers.Dense(output_size,activation=None,use_bias=False)
    
    def call(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


def get_loss(X,Y):
    return tf.reduce_sum(tf.square(Y - X))

neural_net = NeuralNet()
optimizer = tf.optimizers.SGD(learning_rate)

# %% prepare run optimization of Q network

def run_optimization(x,y):
    with tf.GradientTape() as g:
        # forward
        x = neural_net(x)
        # loss
        loss = get_loss(x,y)
    
    trainable_variables = neural_net.trainable_variables;
    gradients = g.gradient(loss,trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# %% do learning

for i in range(num_episodes):
    e = 1./((i * 0.1) + 1)
    rAll = 0
    step_count = 0
    s = env.reset()
    done = False
    
    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        # choose an action by greedily (with chance of e)
        Qs = neural_net(x).numpy()
        if np.random.rand(1) < e :
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
        
        # get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            Qs[0,a] = -100
        else: 
            x1 = np.reshape(s1, [1, input_size])
            # obtain the Q values by feeding the new state through our network
            Qs1 = neural_net(x1).numpy()
            Qs[0,a] = reward + dis * np.max(Qs1)
        
        run_optimization(tf.constant(x), tf.constant(Qs))
        s = s1
    
    rList.append(step_count)
    print("Episode: {} steps: {}".format(i,step_count))
    
    # if last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:])>500:
        break

# %% see our trained network in action
observation  = env.reset()
reward_sum = 0
while True:
    env.render()
    
    x = np.reshape(observation, [1, input_size])
    Qs = neural_net(x).numpy()
    a = np.argmax(Qs)
    
    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("total score: {}".format(reward_sum))
        break

