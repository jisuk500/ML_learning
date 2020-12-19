# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:00:58 2020

@author: jisuk
"""

# https://www.youtube.com/watch?v=qwycq2C-ggY&list=PLQ28Nx3M4Jrguyuwg4xe9d9t2XE639e5C&index=42

# RNN in tensorflow 2.0

# %% import modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)
tf.compat.v1.enable_eager_execution()

# %% one hot encoding for each char in 'hello'

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

#%% one cell RNN input dim(4)
x_data = np.array([[h]], dtype=np.float32)

hidden_size = 2
cell = layers.SimpleRNNCell(units=hidden_size)
rnn = layers.RNN(cell, return_sequences=True, return_state=True)
outputs, states = rnn(x_data)

# equivalent to above
# rnn = layers.SimpleRNN(units=hidden_size, return_sequences=True, return_state=True)
# output, states = rnn(x_data)

print("input only one cell report")
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))


#%% unfolding to n sequences

x_data = np.array([[h, e, l, l, o]],dtype=np.float32)

hidden_size = 2
rnn = layers.SimpleRNN(units=2, return_sequences=True, return_state=True)
outputs, states =  rnn(x_data)

print("-------------------------------------------------------------------------")
print("input wequence data cell report")
print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))


#%% one cell RNN input_dim(4) -> output_dim (2). sequence: 5, batch:3 
# 3 batches 'hello', 'eolll', 'lleel'
x_data = np.array([[h, e, l, l, o],
                   [e, o, l, l, l],
                   [l, l, e, e, l]], dtype=np.float32)

hidden_size = 2
rnn = layers.SimpleRNN(units=2, return_sequences=True, return_state = True)
outputs, states = rnn(x_data)

print('x_data: {}, shape: {}'.format(x_data, x_data.shape))
print('outputs: {}, shape: {}'.format(outputs, outputs.shape))
print('states: {}, shape: {}'.format(states, states.shape))

