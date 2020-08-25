# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:36:07 2020

@author: jisuk
"""

# %% import numpy setup

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

import random
from collections import deque

import gym

env = gym.make('CartPole-v1')
env._max_episode_steps = 10001

# %%constants defining our network model
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.01

dis = 0.99
REPLAY_MEMORY = 50000

# %% neural network constants
n_hidden_1 = 100
n_hidden_2 = 1000
n_hidden_3 = 1000
n_hidden_4 = 1000




# %% setup Neural network details

class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = layers.Dense(n_hidden_1,activation=tf.nn.relu,use_bias=True)
        # self.fc2 = layers.Dense(n_hidden_2,activation=tf.nn.relu,use_bias=True)
        # self.fc3 = layers.Dense(n_hidden_3,activation=tf.nn.relu,use_bias=True)
        # self.fc4 = layers.Dense(n_hidden_4,activation=tf.nn.relu,use_bias=True)
        self.out = layers.Dense(output_size,activation=None)
    
    def call(self,x):
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        x = self.out(x)
        return x
        
    
def get_loss(x,y):
    return tf.reduce_sum(tf.square(y - x))

def train_network(net,x,y):
    with tf.GradientTape() as g:
        # forward
        x = net(x)
        # loss
        l = get_loss(x,y)
        
    trainable_variables = net.trainable_variables
    gradients = g.gradient(l,trainable_variables)
    optimizer.apply_gradients(zip(gradients,trainable_variables))
    
    return l
    

optimizer = tf.optimizers.Adam(learning_rate)
mainDQN = NeuralNet()
resultDQN = NeuralNet()

# %% train from replay buffer

def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0,input_size)
    y_stack = np.empty(0).reshape(0,output_size)
    
    # get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        state = state.reshape([1,-1])
        next_state = next_state.reshape([1,-1])
        
        Q = DQN(state).numpy()
        
        # terminal?
        if done:
            Q[0,action] = reward
        else:
            #obtain the 'Q' values by feeding  the new state through our network
            Q[0,action] = reward + dis * np.max(DQN(next_state).numpy())
            
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
    
    # train out network using target and predicted Q values on each episode
    x_stack = x_stack.astype(np.float32)
    y_stack = y_stack.astype(np.float32)
    
    return train_network(DQN,tf.constant(x_stack),tf.constant(y_stack))

# %% bot play
def bot_play(DQN):
    # see our trained network in action
    s = env.reset()
    s = s.astype(np.float32)
    s2d = s.reshape([1,-1])
    
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(DQN(s2d).numpy())
        s,reward,done,_ = env.step(a)
        s = s.astype(np.float32)
        s2d = s.reshape([1,-1])
        reward_sum+=reward
        
        if done:
            print("Total score: {}".format(reward_sum))
            break

# %% main

def main():
    max_episodes = 2000
    
    # store the previous observations   in replay memory
    replay_buffer = deque()

    
    success_count = 0
    best_loss = 99999999
    global resultDQN
    
    global resultDQN
    for episode in range(max_episodes):
        resultDQN = mainDQN
        e = 1.0/((episode / 10) + 1)
        done = False
        step_count = 0
        
        state = env.reset()
        state  = state.astype(np.float32)
        
        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                # choose an action by greedily from the Q-network
                state2d = state.reshape([1,-1])
                action = np.argmax(mainDQN(state2d).numpy())
            
            
            # get net state and reward from environment
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32)
            
            if done: # big penalty
                reward = -100
            
            # save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()
            
            state = next_state
            step_count += 1
            
            if step_count > 10000: # good enough
                success_count+=1
                break
        
        print("Episode: {}, steps: {}".format(episode, step_count))
        if step_count > 10000:
            pass
        
        if episode % 10 == 1: # train every 10 episodes
            # get a random batch of experiences
            for _ in range(50):
                # minibatch works betters
                minibatch = random.sample(replay_buffer, 10)
                loss = simple_replay_train(mainDQN, minibatch)
            
            print("loss: {}".format(loss))
            
            if best_loss > loss:
                best_loss = loss
                resultDQN.set_weights(mainDQN.get_weights())
            
            
            if success_count >= 9:
                break
            else: 
                print("success count: {}".format(success_count))
                success_count = 0
     
    return mainDQN
# %% do learning
    
mainDQN = main()

# %% do bot play

bot_play(resultDQN)

