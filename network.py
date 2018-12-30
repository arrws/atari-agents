import numpy as np
import tensorflow as tf
import os

from config import *

class Network(object):

    def __init__(self, no_actions, network=None):
        # WEIGHTS
        self.W_conv1 = self.weight_variable([8, 8, 4, 16]) # 20x20x16
        self.b_conv1 = self.bias_variable([16])

        self.W_conv2 = self.weight_variable([4, 4, 16, 32]) # 10x10x32
        self.b_conv2 = self.bias_variable([32])

        self.W_fc1 = self.weight_variable([10*10*32, 256])
        self.b_fc1 = self.bias_variable([256])

        self.W_fc2 = self.weight_variable([256, no_actions])
        self.b_fc2 = self.bias_variable([no_actions])

        # INPUT
        self.s = tf.placeholder(tf.float32, [None, config["frame_w"], config["frame_h"], 4]) # 4 stacked frames
        self.a = tf.placeholder(tf.float32, [None, no_actions]) # actions
        self.y = tf.placeholder(tf.float32, [None]) # target q function

        # LAYERS
        h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4, "SAME") + self.b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2, "SAME") + self.b_conv2)

        h_flat = tf.reshape(h_conv2, [-1, 10*10*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, self.W_fc1) + self.b_fc1)

        # result layer
        self.q = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2 # q action values

        # gradient step
        self.q_value = tf.reduce_sum(tf.multiply(self.q, self.a), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.y - self.q_value))

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.loss, global_step=global_step)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.9, momentum=0.95, epsilon=0.01).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep = 3)

        # param assignment ops only for target network
        if network != None:
            self.copy_ops = []
            self.copy_ops.append(self.W_conv1.assign(network.W_conv1))
            self.copy_ops.append(self.b_conv1.assign(network.b_conv1))
            self.copy_ops.append(self.W_conv2.assign(network.W_conv2))
            self.copy_ops.append(self.b_conv2.assign(network.b_conv2))
            self.copy_ops.append(self.W_fc1.assign(network.W_fc1))
            self.copy_ops.append(self.b_fc1.assign(network.b_fc1))
            self.copy_ops.append(self.W_fc2.assign(network.W_fc2))
            self.copy_ops.append(self.b_fc2.assign(network.b_fc2))


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride, pad):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = pad)


    def copy(self, sess, network):
        sess.run(self.copy_ops)
        # verify
        q_w_fc1 = self.W_fc1.eval(session = sess)
        t_w_fc1 = network.W_fc1.eval(session = sess)
        if np.array_equal(q_w_fc1, t_w_fc1):
            print ("Nice! target network inherited parameter correctly\n")


def restore_network(sess, network):
    # load network if exists
    global step
    checkpoint = tf.train.get_checkpoint_state("tmp")
    if checkpoint and checkpoint.model_checkpoint_path:
        network.saver.restore(sess, checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
        print("Successfully restored:", checkpoint.model_checkpoint_path,"\nglobal step =",step)
    else:
        print("Could not restore network\nglobal step =",step)

