# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:37:44 2020

@author: jisuk
"""

# %% import modules
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

import tensorflow as tf
import numpy as np

# %% MNIST dataset parameters
num_classes = 10  # total classes

# Training parameters.
learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 10

# network parameters
conv1_filters = 32  # number of filters for 1st conv layer
conv2_filters = 64  # number of filters for 2nd conv layer
fc1_units = 1024  # number of neurons for 1st fully-connected

# %% prepare MNIST dta
#from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# normalize images value from [0 255] to [0 1]
x_train, x_test = x_train / 255.0, x_test / 255.

# %% Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# %% create some wrappers for simplicity


def conv2d(x, W, b, strides=1):
    # conv3d wrapper, with bias and relu acrivation.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # maxpool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# %% store layers weight & bias


# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()

weights = {
    # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),
    # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),
    # FC Layer 1: 7*7*64 inputs, 1024 units.
    'wd1': tf.Variable(random_normal([7*7*64, fc1_units])),
    # FC Out Layer: 1024 inputs, 10 units (total number of classes)
    'out': tf.Variable(random_normal([fc1_units, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'out': tf.Variable(tf.zeros([num_classes]))
}
# %% create model


def conv_net(x):

    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolution Layer. Output shape: [-1, 28, 28, 32].
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer. Output shape: [-1, 14, 14, 64].
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    # Fully connected layer, Output shape: [-1, 1024].
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # Apply ReLU to fc1 output for non-linearity.
    fc1 = tf.nn.relu(fc1)

    # Fully connected layer, Output shape: [-1, 10].
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out)

# %% cross-entropy loss function


def cross_entropy(y_pred, y_true):
    # encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# accuracy metric


def accuracy(y_pred, y_true):
    # predicted class is the index of highest score in prediction vector
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# adam optimizer
optimizer = tf.optimizers.Adam(learning_rate)

# %% optimization process


def run_optimization(x, y):
    # wrap computation inside a gradientTape for automatic defferentiation
    with tf.GradientTape() as g:
        pred = conv_net(x)
        loss = cross_entropy(pred, y)

    # variables to update, i.e. trainable vasriables
    trainable_variables = list(weights.values()) + list(biases.values())

    # compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # update W and b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))


# %% Run training for the given number of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # run the optimization to update W and b values
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %% test model on validation set
pred = conv_net(x_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

# %% visualize predictions
# import matplotlib.pyplot as plt

# %% predict 5 images from validation set
n_images = 5
test_images = x_test[:n_images]
predictions = conv_net(test_images)

# display image and model prediction
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i], [28, 28]), cmap='gray')
    plt.show()
    print("Model Prediction: %i" % np.argmax(predictions.numpy()[i]))
