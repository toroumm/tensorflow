


import tensorflow as tf

import numpy as np

import sys

class DeepTF:

	def __init__(self):
		self.learning_rate = 0.01
		self.epochs = 4500
		self.learning_rate = 0.01
		self.momentum = 0
		self.run_epochs = 0
		self.learning_decay = 1.0
		self.learning_decay_batch = 0
		self.reg_lambda = 0
		self.drop_on = 0
		self.drop_threshold = 0.5
		self.batch = 100
		self.train_step = 0
		self.keep_prob = 0
		self.y = 0
		self.y_ = 0
		self.x = 0
		self.inputs = 0
		self.outputs = 0
		self.hidden_layer = 0
		self.net_name = 0
		self.accuracy_train = 0
		self.accuracy_test = 0
		self.test_samples = 0
		self.accuracy_validation = 0
		self.progess = 500

	def set_accuracy_train(self,value):
		self.accuracy_train= value
	def set_accuracy_test(self,value):
		self.accuracy_test = value
	def set_accuracy_validation(self,value):
		self.accuracy_validation = 0
	def set_epochs(self,value):
		self.epochs = value
	def set_learning_rate(self,value):
		self.learning_rate = value
	def set_learning_decay(self,value):
		self.learning_decay = value
	def set_learning_decay_batch(self, value):
		self.learning_decay_batch = value
	def set_batch(self,value):
		self.batch =value
	def set_regularization_l2(self,value):
		self.reg_lambda = value
	def set_momentum(self,value):
		self.momentum =value
	def set_dropout_threshold(self, value):
		self.drop_threshold = value
	def set_batch_step(self, value):
		self.batch = value
	def set_net_name(self,value):
		self.net_name = value
	def set_n_test_samples(self,value):
		self.test_samples = value
	def set_show_progress(self,value):
		self.progress = value


	def get_show_progress(self):
		return self.progress

	def get_hidden_layer(self):
		return self.hidden_layer
	def get_learning_rate(self):
		return self.learning_rate
	def get_momentum(self):
		return self.momentum
	def get_epochs(self):
		return self.epochs
	def get_net_name(self):
		return self.net_name
	def get_learning_decay(self):
		return self.learning_decay
	def get_batch(self):
		return self.batch
	def get_dropout_on(self):
		return self.drop_on
	def get_dropout_threshold(self):
		return self.drop_threshold
	def get_regularization_l2(self):
		return self.reg_lambda
	def get_learning_decay_batch(self):
		return self.learning_decay_batch
	def get_accuracy_test(self):
		return self.accuracy_test
	def get_accuracy_train(self):
		return self.accuracy_train
	def get_accuracy_validation(self):
		return self.accuracy_validation
	def get_n_data_samples(self):
		return self.inputs.shape[0]
	def get_n_validation_samples(self):
		return 0
	def get_n_test_samples(self):
		return self.test_samples


	def build_net(self,hidden_neurons,inputs,outputs):
		self.train_step, self.x, self.y, self.y_, self.keep_prob = self.build_inference(hidden_neurons, inputs, outputs)
	
	def build_inference(self,hidden_neurons, _inputs, _outputs):

		self.inputs = _inputs
		self.outputs = _outputs
		self.hidden_layer = hidden_neurons



		weights = np.empty(hidden_neurons.shape[0] + 1, dtype=tf.Variable)

		bias = np.empty(hidden_neurons.shape[0] + 1, dtype=tf.Variable)

		vjs = np.empty(hidden_neurons.shape[0], dtype=object)

		hidden_drop = np.empty(hidden_neurons.shape[0], dtype=object)

		x = tf.placeholder(tf.float32, [None, self.inputs.shape[1]])

		weights[0] = tf.Variable(tf.truncated_normal([self.inputs.shape[1], hidden_neurons[0]], stddev=0.1))

		bias[0] = tf.Variable(tf.zeros([hidden_neurons[0]]))

		vjs[0] = tf.nn.relu(tf.matmul(x, weights[0]) + bias[0])

		keep_prob = tf.placeholder(tf.float32)

		hidden_drop[0] = tf.nn.dropout(vjs[0], keep_prob)

		for i in xrange(1, hidden_neurons.shape[0]):
			weights[i] = tf.Variable(tf.truncated_normal([hidden_neurons[i - 1], hidden_neurons[i]], stddev=0.1))

			bias[i] = tf.Variable(tf.zeros([hidden_neurons[i]]))

			vjs[i] = tf.nn.relu(tf.matmul(vjs[i - 1], weights[i]) + bias[i])

			hidden_drop[i] = tf.nn.dropout(vjs[i], keep_prob)

		bias[bias.shape[0] - 1] = tf.Variable(tf.zeros(self.outputs.shape[1]))

		weights[weights.shape[0] - 1] = tf.Variable(tf.truncated_normal([hidden_neurons[hidden_neurons.shape[0] - 1], self.outputs.shape[1]], stddev=0.1))

		y = tf.nn.softmax(tf.matmul(vjs[vjs.shape[0] - 1], weights[weights.shape[0] - 1]) + bias[bias.shape[0] - 1])

		y_ = tf.placeholder(tf.float32, [None, self.outputs.shape[1]])

		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		train_step = tf.train.GradientDescentOptimizer(self.get_learning_rate()).minimize(cross_entropy)

		return train_step, x, y, y_, keep_prob

	def run_tfnet(self, test_in, test_out):

		init = tf.initialize_all_variables()

		self.set_n_test_samples(test_in.shape[0])

		sess = tf.Session()

		sess.run(init)

		bt = 0

		for i in xrange(self.epochs):

			batch_xs = self.inputs[bt:bt + self.batch, ]

			batch_ys = self.outputs[bt:bt + self.batch, ]

			bt += self.batch

			if (bt >= (self.inputs.shape[0] - self.batch)):
				bt = 0


			sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: self.drop_threshold})

			correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))

			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

			self.run_epochs = i
			'''
			if (len(val_in) > 0 and i%20):
				accuracy_val = sess.run(accuracy, feed_dict={x: val_in, y_: val_out, keep_prob:1.0})

				if (val_earlier < accuracy_val):
					val_count += 1
					if (val_count >= 5):
						break
				else:
					val_count -= 0.3
			'''

			if (i % self.get_show_progress() == 0):
				print i, sess.run(accuracy, feed_dict={self.x: self.inputs, self.y_: self.outputs})

		self.set_accuracy_test(sess.run(accuracy, feed_dict={self.x: test_in, self.y_: test_out}))
		self.set_accuracy_train(sess.run(accuracy, feed_dict={self.x: self.inputs, self.y_: self.outputs}))



	
