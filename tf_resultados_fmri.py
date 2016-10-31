


import tensorflow as tf

import numpy as np


import sys

def build_inference(hidden_neurons,inputs, outputs):

	weights = np.empty(hidden_neurons.shape[0]+2,dtype=object)

	bias = np.empty(hidden_neurons.shape[0]+1,dtype=object)

	vjs = np.empty(hidden_neurons.shape[0],dtype=object)

	weights[0] = tf.placeholder(tf.float32, [None, inputs])

	weights[1] = tf.Variable(tf.truncated_normal([inputs,hidden_neurons[0]], stddev=0.1))

	bias[0] = tf.Variable(tf.zeros([hidden_neurons[0]]))

	vjs[0] = tf.nn.relu(tf.matmul(weights[0],weights[1]),bias[0])

	for i in xrange(2,hidden_neurons.shape[0]):

		weights[i] = tf.Variable(tf.truncated_normal([hidden_neurons[i-1],hidden_neurons[i]], stddev=0.1))

		bias[i-1] = tf.Variable(tf.zeros([hidden_neurons[i]]))

		vjs[i-1] = tf.nn.relu(tf.matmul(weights[i-1], weights[i]), bias[i-1])

	bias[bias.shape[0]] = tf.Variable(tf.zeros(outputs))

	weights[weights.shape[0]-1] = tf.Variable(tf.truncated_normal([hidden_neurons[hidden_neurons.shape[0]-1],outputs], stddev=0.1))


