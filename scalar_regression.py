#!/usr/bin/env python

import tensorflow as tf
import numpy as np

# how large can our weight updates be?
LEARNING_RATE = 0.01

# size of our dataset
N = 1000

# training data
# X is uniformly IID in [0, 1)
# Y is a linear function of X + some noise
X = np.random.uniform(size=N)
Y = (X * 10 + 1) + np.random.uniform(low=-1, high=1, size=N)

# our training placeholders
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')

# The simplest gradient model we can make.
# A scalar linear function of one weight and one bias.
weight = tf.Variable(0.1, name='weight')
bias = tf.Variable(0.0, name='bias')
y_p = weight * x + bias

# How bad was our prediction?  Use the L2 loss of our
# prediction vs our label.
loss = (y_p - y) ** 2

# for printing progress, our average loss
loss_mean = tf.reduce_mean(loss, name='mean_loss')

# Update our weights to minimize our cost function:
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
train_op = optimizer.minimize(loss)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(10):
        for i in range(N):
            sess.run(train_op, { x: X[i], y: Y[i] })

        print '%d %f %f %f' % (
            epoch, sess.run(loss_mean, { x: X, y: Y }),
            sess.run(weight), sess.run(bias))
