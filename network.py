import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from config import *

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



def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder



class AC_Network(object):

    def __init__(self, no_actions, scope=None, trainer=None):

        with tf.variable_scope(scope):
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

            # LAYERS
            h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4, "SAME") + self.b_conv1)
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2, "SAME") + self.b_conv2)
            h_flat = tf.reshape(h_conv2, [-1, 10*10*32])


            self.policy = slim.fully_connected(h_flat, no_actions,
                activation_fn=tf.nn.softmax,
                weights_initializer=self.normalized_columns_initializer(0.01),
                biases_initializer=None)

            self.value = slim.fully_connected(h_flat, 1,
                activation_fn=None,
                weights_initializer=self.normalized_columns_initializer(1.0),
                biases_initializer=None)

        if scope != 'global':
            self.a = tf.placeholder(tf.float32, [None, no_actions]) # actions
            self.y = tf.placeholder(tf.float32, [None]) # target v function


            self.log_prob = tf.log( tf.reduce_sum(self.policy * self.a, axis=1, keep_dims=True) + 1e-10)

            self.advantage = self.y - self.value

            self.loss_policy = - self.log_prob * tf.stop_gradient(self.advantage)
            self.loss_value  = config["loss_v"] * tf.square(self.advantage)
            self.entropy = config["loss_entropy"] * tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=1, keep_dims=True)
            self.loss = tf.reduce_mean(self.loss_policy + self.loss_value + self.entropy)


            global_step = tf.Variable(0, name='global_step', trainable=False)

            self.optimizer = tf.train.RMSPropOptimizer(5e-3, decay=.99).minimize(self.loss)

            self.saver = tf.train.Saver(max_to_keep = 3)

            #Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            self.gradients = tf.gradients(self.loss,local_vars)
            self.var_norms = tf.global_norm(local_vars)
            grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)

            #Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


    def normalized_columns_initializer(self, std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride, pad):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = pad)


