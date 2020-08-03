# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:11:48 2020

@author: jisuk
"""
# %% import the modules
from __future__ import absolute_import, division, print_function

from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

# %% MNIST dataset parameter
num_classes = 10  # total classes (0~9 digits)
num_features = 784  # data features (img shape is 28 * 28)

# training data
learning_rate = 0.1
training_steps = 2000
batch_size = 256
display_step = 100

# Network parameters
n_hidden_1 = 128  # 1st layer number of neurons.
n_hidden_2 = 256  # 2nd layer number of neurons.

# %% Prepare MNIST data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28 * 28)
x_train, x_test = x_train.reshape(
    [-1, num_features]), x_test.reshape([-1, num_features])
# normalize images value from [0 255] to [0 1]
x_train, x_test = x_train/255., x_test/255.

# %% Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# %% Create TF Model


class NeuralNet(Model):
    # set layers
    def __init__(self):
        super(NeuralNet, self).__init__()
        # first fully-connected hidden layer
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)
        # Second fully-connect hidden layer
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
        # output fully-connected layer
        self.out = layers.Dense(num_classes)

    # Set forward pass
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without soffmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Build neural network model.
neural_net = NeuralNet()

# %% cross entropy Loss

# note that this will apply softmax to the digits


def cross_entropy_loss(x, y):
    # convert labels to int64 for tf cross-entropy fuction
    y = tf.cast(y, tf.int64)
    # apply softmax to logits and compute cross-entropy
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch
    return tf.reduce_mean(loss)

# Accuracy metric


def accuracy(y_pred, y_true):
    # predicted class is the index of highest score in prediction vector(i,e, argmax)
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Stochastic gradient desecnt optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# %% optimization process


def run_optimization(x, y):
    # Wrap compare inside a gradient tape for automatic differentiation
    with tf.GradientTape() as g:
        # Forward pass.
        pred = neural_net(x, is_training=True)
        # compute loss
        loss = cross_entropy_loss(pred, y)

    # Variable to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables

    # compute gradients
    gradients = g.gradient(loss, trainable_variables)

    # update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# %% run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # run the optimization to update W and b values
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %% test model on validation set
pred = neural_net(x_test, is_training=False)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %% visualize predictions
import matplotlib.pyplot as plt

# %% predict 5 image from valiation set
n_images = 30
test_images = x_test[:n_images]
test_answers = y_test[:n_images]
predictions = neural_net(test_images)

# display image and model prediction

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model Prediction: %i, answer: %i" % (np.argmax(predictions.numpy()[i]), test_answers[i]))
