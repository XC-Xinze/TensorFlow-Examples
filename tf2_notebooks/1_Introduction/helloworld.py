# %%
"""
# Hello World

A very simple "hello world" using TensorFlow v2 tensors.

- Author: Aymeric Damien
- Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# %%
import tensorflow as tf

# %%
# Create a Tensor.
hello = tf.constant("hello world")
print(hello)

# %%
# To access a Tensor value, call numpy().
print(hello.numpy())
