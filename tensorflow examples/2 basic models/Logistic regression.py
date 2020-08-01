# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:11:42 2020

@author: jisuk
"""

# %% import basic modules
from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

import tensorflow as tf
import numpy as np

# %% MNIST dataset parameters
num_classes = 10  # 0~9 digits
num_features = 784  # 28 * 28

# training parameters
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50

# %% prepare MNIST data

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert to float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float16)
# Flatten image to 1-D vector of 784 features (28*28)
x_train, x_test = x_train.reshape(
    [-1, num_features]), x_test.reshape([-1, num_features])
# normalize images vaue from [0,255] to [0 1]
x_train, x_test = x_train/255., x_test / 255.

# %%Use tf.data API to shuffle and batch data
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# %%weight of shape [784,10] the 28 * 28 image features and total number of classes
W = tf.Variable(tf.ones([num_features, num_classes]), name='weight')
# Bias of shape [10] , the tota number of classes
b = tf.Variable(tf.zeros([num_classes]), name='bias')

# logistic regression (Wx + b)


def logistic_regression(x):
    # apply softmax to normalize the logits to a probability distribution
    return tf.nn.softmax(tf.matmul(x, W)+b)

# cross-Entropy loss function


def cross_entropy(y_pred, y_true):
    # encode label to a one-hot vector
    y_true = tf.one_hot(y_true, depth=num_classes)
    # clip prediction values to aviod log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred), 1))

# accuracy metric


def accuracy(y_pred, y_true):
    # predicted class is the index of highest score in prediction vector(i.e. argmax)
    correct_prediction = tf.equal(
        tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# stochastic gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# %% Optimization process


def run_optimization(x, y):
    # wrap conputation inside a gradientTape for automatic differetiation
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)

    # compute gradients
    gradients = g.gradient(loss, [W, b])

    # update W and b following gradients
    optimizer.apply_gradients(zip(gradients, [W, b]))
# %% Run training for the given number of steps


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # run the oprimization to update W and b values
    run_optimization(batch_x, batch_y)

    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# %% test model on validation  set
x_test_tensor = tf.constant(x_test, dtype=tf.float32)
pred = logistic_regression(x_test_tensor)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# %% visualize predictions
import matplotlib.pyplot as plt

# %%predict 5 images from validation set

prediction_count = 20
n_images = np.random.randint(0,x_test.shape[0],(prediction_count))
test_images = x_test[n_images]
test_y_answer = y_test[n_images]
test_images_tensor = tf.constant(test_images, dtype=tf.float32)
predictions = logistic_regression(test_images_tensor)
# display image and model prediction
for i in range(prediction_count):
    showimg = np.reshape(test_images[i], [28, 28])
    showimg = np.uint8(showimg * 255)
    plt.imshow(np.reshape(showimg, [28, 28]), cmap='gray')
    plt.show()
    print("Model prediction: %i, Answer: %i" % (np.argmax(predictions.numpy()[i]),test_y_answer[i]))
