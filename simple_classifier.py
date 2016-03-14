#!/usr/bin/env python

import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.1
N = 10000
BATCH_SIZE = 100
DIM = 3

# training data
# X is a NxD matrix of random numbers
# Y is a Nx2, non-linear function of X: is X contained inside a sphere of radius 1?
X = np.random.uniform(low=-1, high=1, size=N*DIM).reshape((N, DIM))
Y = np.zeros((N, 2))
Y[:,0] = ((X ** 2).sum(axis=1) > 1.0).astype(np.float)
Y[:,1] = ((X ** 2).sum(axis=1) <= 1.0).astype(np.float)

# Placeholder variables we will fill with our training data.
x = tf.placeholder(tf.float32, name='x', shape=[None, 3])
y = tf.placeholder(tf.float32, name='y')

# We'll use RELUs as our activation function between layers:
# https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
def activation(x):
    return tf.nn.relu(x)

# A fully connected layer: activation(matrix-multiply + bias)
def layer(name, x, num_inputs, num_outputs):
    with tf.name_scope(name) as scope:
        weight = tf.Variable(np.random.uniform(size=(num_inputs, num_outputs)).astype(np.float32), name='weight')
        bias = tf.Variable(0.0, name='bias')
        return activation(tf.matmul(x, weight) + bias), weight, bias

# slap a few layers together
y_1, w1, b1 = layer('layer1', x, DIM, 25)
y_2, w2, b2 = layer('layer2', y_1, 25, 25)
y_3, w3, b3 = layer('layer3', y_2, 25, 10)
y_4, w4, b4 = layer('layer4', y_3, 10, 2)

# convert the output of our layers to probabilities, and
# measure the loss vs. the "true" probabilities (our labels)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_4, y)
loss = tf.reduce_mean(cross_entropy, name='mean_loss')

# Update our weights to minimize our cost function:
#optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
#optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, 1e-3)
#optimizer = tf.train.AdamOptimizer(LEARNING_RATE, 1e-3)
optimizer = tf.train.AdagradOptimizer(LEARNING_RATE)
train_op = optimizer.minimize(loss)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(100):
        print '%d %s %s' % (epoch,
                            sess.run(w4).mean(),
                            sess.run(loss, { x: X[0:BATCH_SIZE], y: Y[0:BATCH_SIZE] }))
        for i in range(0, N, BATCH_SIZE):
            sess.run(train_op, { x: X[i:i+BATCH_SIZE], y: Y[i:i+BATCH_SIZE] })

