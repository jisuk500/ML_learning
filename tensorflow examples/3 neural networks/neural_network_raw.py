# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:15:11 2020

@author: jisuk
"""

# %% immport some modules
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
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
    'h1': tf.Variable(random_normal([num_features, n_hidden_1])),
    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

# %% create model


def neural_net(x):
    # Hidden fully connected layer with 128 neurons.
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Apply sigmoid to layer_1 output for non-linearity.
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons.
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Apply sigmoid to layer_2 output for non-linearity.
    layer_2 = tf.nn.sigmoid(layer_2)

    # Output fully connected layer with a neuron for each class.
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out_layer)


# stochastic gradient descent
optimizer = tf.optimizers.SGD(learning_rate)

# %% cross entropy loss function

def cross_entropy(y_pred, y_true):
    # encode label to a one hot encoder
    y_true = tf.one_hot(y_true, depth=num_classes)
    # clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# accuracy function


def accuracy(y_pred, y_true):
    # predicted class is the index of highest score in prediction vector (i.e. argmax)
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learning_rate)


# %% optmization process

def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x)
        loss = cross_entropy(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# %% Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %% test model on validation set.
pred = neural_net(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# visualize predictions.
import matplotlib.pyplot as plt

# predict 5 images from validation set.
n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

# display image and model predictions
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i" % np.argmax(predictions.numpy()[i]))
