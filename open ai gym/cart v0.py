# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:10:46 2020

@author: jisuk
"""

# %% import modules

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Conv2D, BatchNormalization, Activation

import random
from collections import deque

import gym

# %% initialize modules
env = gym.make("Breakout-v0")

# %% neural net constants
history_size = 4
input_size = env.observation_space.shape
input_size = (input_size[0], input_size[1],history_size)
output_size = env.action_space.n
lr = 0.001

k1 = 16
k2 = 32
k3 = 32
fc1 = 256

# %% gym game constants

dis = 0.99
REPLAY_MEMORY = 20000

# %% define network class

def NeuralNet():
    model = Sequential()


    model.add(Conv2D(k1, (3, 3), activation=None, padding='same',
                     input_shape=input_size, name='InputLayer'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2,2)))


    model.add(Conv2D(k2, (3, 3), activation=None,
                      padding='same', name='HiddenLayer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D((2, 2)))


    # model.add(Conv2D(k3, (3, 3), activation=None,
    #                   padding='same', name='HiddenLayer2'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))
    # model.add(MaxPooling2D((2, 2)))


    model.add(Flatten())
    model.add(Dense(fc1, activation='relu', name='FcLayer'))
    model.add(Dense(output_size, activation=None, name='OutputLayer'))


    opt = optimizers.Adam(learning_rate=lr)

    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model

mainDQN = NeuralNet()
mainDQN.summary()
targetDQN = NeuralNet()

targetDQN.set_weights(mainDQN.get_weights())

# %% DQN methods


def replay_train(mainDQN_, targetDQN_, train_batch):

    x_stack = np.zeros(0).reshape([0] + list(input_size))
    y_stack = np.zeros(0).reshape(0,output_size)

    # get stored info from the buffer
    for history_state, action, reward, next_history_state, done in train_batch:
        history_state2d = history_state.reshape([1] + list(input_size))
        history_state2d = history_state2d * 1.0/255.0
        history_state2d = history_state2d.astype(np.float32)

        next_history_state2d = next_history_state.reshape([1] + list(input_size))
        next_history_state2d = next_history_state2d * 1.0/255.0
        next_history_state2d = next_history_state2d.astype(np.float32)
        
        Q = mainDQN(history_state2d).numpy()
        
        # terminal?
        if done:
            Q[0,action] = reward
        else:
            
            
            Q[0,action] = reward + dis *  np.max(targetDQN_(next_history_state2d).numpy())
        
        x_stack = np.vstack([x_stack, history_state2d])
        y_stack = np.vstack([y_stack, Q])
    
    # train out network using target and predicted Q values on each episode
    x_stack = x_stack.astype(np.float32)
    y_stack = y_stack.astype(np.float32)
    
    return mainDQN_.fit(x_stack,y_stack)

        
def one_hot(x):
    return np.identity(6)[x:x+1]


# %% function for atari game
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype(np.uint8)
    gray = gray.reshape(list(gray.shape)+[-1])

    return gray

def initialize_state_history(initState):
    grayimg = rgb2gray(initState)
    hist = np.zeros(0).reshape(210,160,0)
    
    for i in range(4):
        hist = np.concatenate((hist,grayimg),axis=2)
        
    return hist

def update_state_history(histState, newState):
    histState = np.delete(histState,0,2)
    grayimg = rgb2gray(newState)
    histState = np.concatenate((histState,grayimg),axis=2)
    return histState
    

# %%

def main():
    max_episodes = 100000
    
    #store the previous observations in replay memory
    replay_buffer = deque()
    
    best_loss = 99999999
    global targetDQN
    global mainDQN
    
    for episode in range(max_episodes):
        e = 1.0/((episode / 100) + 1)
        done = False
        step_count = 0
        last_reward = 0

        state = env.reset()
        
        env.render()
        
        history_state = initialize_state_history(state)

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                # choose an action by greedily from the Q-network
                history_state2d = history_state.reshape([1] + list(input_size))
                history_state2d = history_state2d * 1.0/255.0
                history_state2d = history_state2d.astype(np.float32)
                action = np.argmax(mainDQN(history_state2d).numpy())

            # get net state and reward from environment
            next_state, reward, done, _ = env.step(action)
            next_history_state = update_state_history(history_state, next_state)
            
            env.render()

            # save the experience to our buffer
            replay_buffer.append((history_state, action, reward, next_history_state, done))
            if len(replay_buffer) > REPLAY_MEMORY:
                replay_buffer.popleft()

            state = next_state
            history_state = next_history_state
            
            step_count += 1
            last_reward += reward

        print("Episode: {}, steps: {}, score: {}".format(episode, step_count, last_reward))

        if episode % 100 == 1:  # train every 100 episodes
            # get a random batch of experiences
            for _ in range(10):
                # minibatch works betters
                minibatch = random.sample(replay_buffer, 100)
                hist = replay_train(mainDQN, targetDQN, minibatch)
                loss = hist.history['loss']
                acc = hist.history['accuracy']

            targetDQN.set_weights(mainDQN.get_weights())
            print("loss: {}".format(loss[0]))

            if best_loss > loss[0]:
                best_loss = loss[0]
                tf.keras.models.save_model(targetDQN,"targetDQN.h5")
                print("saved new!")

    return mainDQN

# %% run main

main()

# %% 

