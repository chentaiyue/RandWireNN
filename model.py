import tensorflow as tf
from dag import DAGLayer
from Graph import *
from config import cfg
from sep_conv import Sep_conv
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


class RandWireNN(object):
	def __init__(self, sess):
		self.global_step = tf.Variable(0, False, name='global_step')
		learning_rate = tf.train.exponential_decay(
			learning_rate=cfg.learning_rate,
			global_step=self.global_step,
			decay_steps=cfg.decay_steps,
			decay_rate=cfg.decay_rate,
			staircase=cfg.staircase,
			name='exponential_decay_learning_rate'
		)
		learning_rate = tf.maximum(learning_rate, 1e-5)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.input_img = tf.placeholder(
			dtype=tf.float32,
			shape=[None, cfg.img_size, cfg.img_size, cfg.img_channel],
			name='input_img')
		self.channel = cfg.channel
		if cfg.training:
			self.input_label = tf.placeholder(
				dtype=tf.float32,
				shape=[None, cfg.num_labels]
			)
			
		if cfg.random_graph.lower() == 'ws':
			self.edges = ws(cfg.num_nodes, cfg.K, cfg.prob)
		elif cfg.random_graph.lower() == 'ba':
			self.edges = ba(cfg.num_nodes, cfg.M)
		elif cfg.random_graph.lower() == 'er':
			self.edges = er(cfg.num_nodes, cfg.prob)
		
		self.sess = sess
	
	def bulid(self, input_tensor):
		# self.conv1 = Sep_conv(input_tensor,cfg.img_channel,self.channel//2,stride=[2,2],padding='SAME')
		self.conv1 = tf.layers.conv2d(input_tensor, self.channel, self.channel, [2, 2], 'SAME')
		self.relu1 = tf.nn.relu(self.conv1)
		self.dag1 = DAGLayer(self.relu1, self.channel, self.channel*2, cfg.num_nodes, self.edges)
		self.flatten = tf.layers.flatten(self.dag1)
		
		self.fc = tf.nn.softmax(tf.layers.Dense(cfg.num_labels)(self.flatten), axis=-1)
		return self.fc
	
	def loss(self, logit, label):
		return -tf.reduce_mean(label * tf.log(logit) + (1.0 - label) * tf.log(1.0 - logit))
	
	def train(self):
		mnist = input_data.read_data_sets(r'',
		                                  one_hot='true')
		logits = self.bulid(self.input_img)
		Loss = self.loss(logits, self.input_label)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = self.optimizer.minimize(Loss, self.global_step)
		
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(self.input_label, 1)), tf.float32))
		self.sess.run(tf.global_variables_initializer())
		
		for i in range(1000):
			img, lab = mnist.train.next_batch(100)
			img = np.reshape(img, [100, 28, 28, 1])
			acc, _, l = self.sess.run([accuracy, train_op, Loss],
			                          {self.input_img: img, self.input_label: lab})
			print(l, acc)
		s = 0.0
		for i in range(100):
			img, lab = mnist.test.next_batch(100)
			img = np.reshape(img, [100, 28, 28, 1])
			acc = self.sess.run(accuracy, {self.input_img: img, self.input_label: lab})
			s += acc
		
		print(s / 100.0)


with tf.Session() as sess:
	net = RandWireNN(sess)
	net.train()