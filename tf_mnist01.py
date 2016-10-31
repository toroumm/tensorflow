

from tensorflow.examples.tutorials.mnist import input_data

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




#mnist = input_data.read_data_sets("MNIST_data/",one_hot =True)

path = '/home/jeferson/Dropbox/folds_fmri_cocaina/'

base = 5
fold = 1


data_in = np.loadtxt(path+'base_'+str(base)+'_fmri_treino_fold'+str(fold),dtype=float)

data_out = np.loadtxt(path+'base_'+str(base)+'_fmri_treino_out_fold'+str(fold),dtype=int)

teste_in = np.loadtxt(path+'base_'+str(base)+'_fmri_teste_fold'+str(fold),dtype=float)

teste_out = np.loadtxt(path+'base_'+str(base)+'_fmri_teste_out_fold'+str(fold),dtype=int)


size_input = data_in.shape[1]

size_ouput = data_out.shape[1]

#data_in = (data_in - np.min(data_in)) / (np.max(data_in) - np.min(data_in))


#x = tf.placeholder(tf.float32,[None,784])

#w = tf.Variable(tf.zeros([784,10]))

#b = tf.Variable(tf.zeros([10]))


neurons = 100

x = tf.placeholder(tf.float32,[None,size_input])

w = tf.Variable(tf.truncated_normal([size_input,neurons], stddev=0.1))

b = tf.Variable(tf.zeros([neurons]))

hidden1 = tf.nn.relu(tf.matmul(x,w)+b)

w2 = tf.Variable(tf.truncated_normal([neurons,size_ouput],stddev = 0.1))

b2 = tf.Variable(tf.zeros([size_ouput]))

y = tf.nn.softmax(tf.matmul(hidden1,w2)+b2)

y_ = tf.placeholder(tf.float32, [None,size_ouput])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

bt = 0

bt_step = data_in.shape[0]

for i in xrange(1000):


	#batch_xs, batch_ys = mnist.train.next_batch(100)


	batch_xs = data_in[bt:bt+bt_step,]

	batch_ys = data_out[bt:bt+bt_step,]


	bt += bt_step;

	if(bt >= (data_in.shape[0]-bt_step)):
		bt = 0



	sess.run(train_step,feed_dict = {x: batch_xs, y_:batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	#print(sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

	print(sess.run(accuracy, feed_dict={x: data_in, y_: data_out}),sess.run(accuracy, feed_dict={x: teste_in, y_: teste_out}))
