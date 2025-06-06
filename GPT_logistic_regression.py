# %%
"""
Logistic Regression (TensorFlow 2.x version)

This script implements a simple logistic regression on MNIST, rewritten from a 
TensorFlow 1.x example to TensorFlow 2.x. Key changes include:

1. Eager execution replaces tf.placeholder / tf.Session.
2. MNIST data is loaded via tf.keras.datasets.mnist.
3. Model parameters (W, b) are tf.Variable and automatically initialized.
4. Gradients are computed with tf.GradientTape and applied with a Keras optimizer.
5. Batching is handled by tf.data.Dataset instead of next_batch.
6. A small epsilon is added inside tf.math.log to avoid log(0).
"""

import tensorflow as tf
import numpy as np

# ------------------------------------------------------------------------------
# 1. Load and preprocess MNIST
# ------------------------------------------------------------------------------
# Use tf.keras.datasets.mnist.load_data() to get (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape images from (N, 28, 28) to (N, 784), cast to float32 and normalize to [0, 1]
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 784).astype("float32") / 255.0

# Convert integer labels to one-hot vectors of length 10
y_train = tf.one_hot(y_train, depth=10)
y_test  = tf.one_hot(y_test, depth=10)

# ------------------------------------------------------------------------------
# 2. Set hyperparameters
# ------------------------------------------------------------------------------
learning_rate    = 0.01
training_epochs  = 25
batch_size       = 100
display_step     = 1

# ------------------------------------------------------------------------------
# 3. Define model parameters W and b (tf.Variable is initialized immediately)
# ------------------------------------------------------------------------------
W = tf.Variable(tf.zeros([784, 10]), name="weights")
b = tf.Variable(tf.zeros([10]), name="bias")

# ------------------------------------------------------------------------------
# 4. Define the forward model (softmax) and the loss function (cross-entropy)
# ------------------------------------------------------------------------------
@tf.function
def model(x):
    """
    Forward pass: compute softmax(Wx + b).
    Returns a tensor of shape (batch_size, 10) representing predicted probabilities.
    """
    logits = tf.matmul(x, W) + b
    return tf.nn.softmax(logits)

def loss_fn(x, y):
    """
    Compute average cross-entropy loss for a batch.
    y is already one-hot encoded. 
    A small epsilon is added inside log to prevent log(0).
    """
    epsilon = 1e-8
    y_pred = model(x)
    cross_entropy = -tf.reduce_sum(y * tf.math.log(y_pred + epsilon), axis=1)
    return tf.reduce_mean(cross_entropy)

# ------------------------------------------------------------------------------
# 5. Define the optimizer (use Keras optimizer instead of tf.train.GradientDescentOptimizer)
# ------------------------------------------------------------------------------
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# ------------------------------------------------------------------------------
# 6. Build the tf.data.Dataset pipeline for training
# ------------------------------------------------------------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
                              .shuffle(buffer_size=60000) \
                              .batch(batch_size)

# ------------------------------------------------------------------------------
# 7. Training loop
# ------------------------------------------------------------------------------
for epoch in range(training_epochs):
    avg_cost = 0.0
    total_batch = int(x_train.shape[0] / batch_size)

    # Iterate over all batches in this epoch
    for batch_x, batch_y in train_dataset:
        # Compute gradients with GradientTape
        with tf.GradientTape() as tape:
            loss_value = loss_fn(batch_x, batch_y)
        # Get gradients of loss w.r.t. [W, b]
        grads = tape.gradient(loss_value, [W, b])
        # Update parameters
        optimizer.apply_gradients(zip(grads, [W, b]))
        # Accumulate average loss
        avg_cost += loss_value.numpy() / total_batch

    # Every 'display_step' epochs, print cost
    if (epoch + 1) % display_step == 0:
        print("Epoch: {:04d} cost={:.9f}".format(epoch + 1, avg_cost))

print("Optimization Finished!")

# ------------------------------------------------------------------------------
# 8. Evaluate accuracy on a subset of the test data (first 3000 examples)
# ------------------------------------------------------------------------------
x_test_subset = x_test[:3000]
y_test_subset = y_test[:3000]

# 1) Compute predictions (softmax probabilities) for test subset
predictions = model(x_test_subset)

# 2) Compare predicted class (argmax) to true class (argmax), yielding a boolean tensor
correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(y_test_subset, axis=1))

# 3) Cast booleans to float32 and take mean: this is the accuracy
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

print("Accuracy:", accuracy.numpy())

