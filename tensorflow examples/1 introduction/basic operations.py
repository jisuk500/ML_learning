# %%
from __future__ import print_function

import tensorflow as tf
import numpy as np

# %%
# Define tensor values
a = tf.constant(2)
b = tf.constant(3)
c = tf.constant(5)
# %%
# various tensor operations
# note : tensor also support python operators(+,*,...)
add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

# %%
# access tensors value
print("add:", add.numpy())
print("sub:", sub.numpy())
print("mul:", mul.numpy())
print("div:", div.numpy())

# %%
# some more operations
mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])

# %%
# access tensors value
print("mean=", mean.numpy())
print("sum=", sum.numpy())

# %%
# matrix multiplications
matrix1 = tf.constant([[1., 2.], [3., 4.]])
matrix2 = tf.constant([[5., 6.], [7., 8.]])

product = tf.matmul(matrix1, matrix2)

# %%
# Display Tensor.
product

# %%
# Convert Tensor to Numpy.
product.numpy()
