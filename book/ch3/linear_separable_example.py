# A simple implementation that learns to classify 2 linearly separable classes.
# The data in each class is generated via a random, controlled process.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_samples_per_class = 1000

a_class = 0
a_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

b_class = 1
b_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

inputs = np.vstack((a_samples, b_samples)).astype(np.float32)

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# Let's visualise the data once
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


def model(inputs):
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    per_sample_loss = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_loss)


learning_rate = 1e-1


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        next_predictions = model(inputs)
        loss = square_loss(targets, next_predictions)
    d_by_dW, d_by_db = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * d_by_dW)
    b.assign_sub(learning_rate * d_by_db)
    return loss


for step in range(40):
    loss_at_step = training_step(inputs, targets)
    print(f"Loss at step {step} = {loss_at_step:.4f}")

predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
