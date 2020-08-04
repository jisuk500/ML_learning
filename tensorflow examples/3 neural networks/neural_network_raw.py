# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:15:11 2020

@author: jisuk
"""

# %% immport some modules
from __future__ import absolute_import, division, print_function
from tensorflow.keras.datasets import mnist

import tensorflow as tf
import numpy as np

# %% NMIST dataset parameters

num_classes = 10  # total classes (0~9) digits
num_features = 784  # data features(img shape = 28 * 28)

# training parameters
learning_rate = 0.001
training_steps = 3000
batch_size = 256
display_step = 100

# network parameters
n_hidden_1 = 128
n_hidden_2 = 256

# %% prepare MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# flatten images to 1_D vector of 784 features (28*28)
x_train, x_test = x_train.reshape(
    [-1, num_features]), x_test.reshape([-1, num_features])
# normalize images value from [0, 255] to [0, 1]
x_train, x_test = x_train/255.0, x_test/255.0

# %% Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# %% store layers weight & bias

# a random value generator to imitialize weights
random_normal = tf.initializers.RandomNormal()

weights = {
    'h1': tf.Variable((random_normal([num_features, n_hidden_1]))),
    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([num_classes])),
    'out': tf.variable(tf.zeros([num_classes]))
}

# %% create model


def neural_net(x):
    # hidden fully connected layer with 128 neurons.
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden fully connected layer with 256 neurons.
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # apply sigmoid to layer_2 output for non-linearity.
    layer_2 = tf.nn.sigmoid(layer_2)

    # output fully connected layer with a neuron for each class.
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # apply softmax to normalize the logits to a probablity distribution.
    return tf.nn.softmax(out_layer)


# stochastic gradient descent
optimizer = tf.optimizers.SGD(learning_rate)

# %% optmization process
