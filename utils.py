
import tensorflow as tf
import numpy as np
import gym
from gym import envs
import cv2

class DQNetwork(object):

    def __init__(self, frame_size, no_actions, learn_rate, network):
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
        self.s = tf.placeholder(tf.float32, [None, frame_size, frame_size, 4]) # 4 stacked frames
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

    # def __init__(self, frame_size, no_actions, learn_rate, network):
    #
    #     # WEIGHTS
    #     self.W_conv1 = self.weight_variable([8, 8, 4, 32]) # 20x20x32
    #     self.b_conv1 = self.bias_variable([32])
    #
    #     self.W_conv2 = self.weight_variable([4, 4, 32, 64]) # 9x9x64
    #     self.b_conv2 = self.bias_variable([64])
    #
    #     self.W_conv3 = self.weight_variable([3, 3, 64, 64]) # 7x7x64
    #     self.b_conv3 = self.bias_variable([64])
    #
    #     self.W_fc1 = self.weight_variable([7*7*64, 512]) # 512
    #     self.b_fc1 = self.bias_variable([512])
    #
    #     self.W_fc2 = self.weight_variable([512, no_actions])
    #     self.b_fc2 = self.bias_variable([no_actions])
    #
    #     # INPUT
    #     self.s = tf.placeholder(tf.float32, [None, frame_size, frame_size, 4]) # 4 stacked frames
    #     self.a = tf.placeholder(tf.float32, [None, no_actions]) # actions
    #     self.y = tf.placeholder(tf.float32, [None]) # target q function
    #
    #     # LAYERS
    #     h_conv1 = tf.nn.relu(self.conv2d(self.s, self.W_conv1, 4, "SAME") + self.b_conv1)
    #     h_conv2 = tf.nn.relu(self.conv2d(h_conv1, self.W_conv2, 2, "VALID") + self.b_conv2)
    #     h_conv3 = tf.nn.relu(self.conv2d(h_conv2, self.W_conv3, 1, "VALID") + self.b_conv3)
    #
    #     h_flat = tf.reshape(h_conv3, [-1, 7*7*64])
    #     h_fc1 = tf.nn.relu(tf.matmul(h_flat, self.W_fc1) + self.b_fc1)
    #
    #     # result layer
    #     self.q = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2 # q action values
    #
    #     # gradient step
    #     self.q_value = tf.reduce_sum(tf.multiply(self.q, self.a), reduction_indices=1)
    #     self.loss = tf.reduce_mean(tf.square(self.y - self.q_value))
    #
    #     global_step = tf.Variable(0, name='global_step', trainable=False)
    #     # self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, global_step=global_step)
    #     self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.9, momentum=0.95, epsilon=0.01).minimize(self.loss)
    #
    #     self.saver = tf.train.Saver(max_to_keep = 3)
    #
    #     # param assignment ops only for target network
    #     if network != None:
    #         self.copy_ops = []
    #         self.copy_ops.append(self.W_conv1.assign(network.W_conv1))
    #         self.copy_ops.append(self.b_conv1.assign(network.b_conv1))
    #         self.copy_ops.append(self.W_conv2.assign(network.W_conv2))
    #         self.copy_ops.append(self.b_conv2.assign(network.b_conv2))
    #         self.copy_ops.append(self.W_conv3.assign(network.W_conv3))
    #         self.copy_ops.append(self.b_conv3.assign(network.b_conv3))
    #         self.copy_ops.append(self.W_fc1.assign(network.W_fc1))
    #         self.copy_ops.append(self.b_fc1.assign(network.b_fc1))
    #         self.copy_ops.append(self.W_fc2.assign(network.W_fc2))
    #         self.copy_ops.append(self.b_fc2.assign(network.b_fc2))


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride, pad):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = pad)

    # def max_pool(self, x):
    #     return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding = "VALID")

    def evaluate(self, sess, state):
        return self.q.eval(session = sess, feed_dict = {self.s: state})

    def train(self, sess, action_list, state_list, target_reward_list):
        sess.run(self.optimizer, feed_dict = {self.a: action_list,
                                         self.s: state_list,
                                         self.y: target_reward_list})

    def copy(self, sess, network):
        sess.run(self.copy_ops)
        # verify
        q_w_fc1 = self.W_fc1.eval(session = sess)
        t_w_fc1 = network.W_fc1.eval(session = sess)
        if np.array_equal(q_w_fc1, t_w_fc1):
            print ("Nice! target network inherited parameter correctly\n")



class StepCounter():
    def __init__(self):
        self.step_counter = 0
        self.max_thread_step = 999999999
    def plus(self):
        self.step_counter += 1
    def get(self):
        return self.step_counter
    def get_max_step(self):
        return self.max_thread_step
    def set(self, x):
        self.step_counter = x

# no short ep
# class BreakoutWrapper():
#     def __init__(self):
#         self.name = 'Breakout-v0'
#         self.env = gym.make(self.name)
#         self.game_offset = 1
#         self.no_actions = 3
#         self.lives = 5 #info.get('ale.lives')
#         print("valid actions:", self.no_actions, self.env.env.get_action_meanings(),'\n')
#
#     def step(self, action_index):
#         f, r, done, info = self.env.step(action_index + self.game_offset)
#         r = np.clip(r, -1, 1)
#         if self.lives > info.get('ale.lives'):
#             self.lives = info.get('ale.lives')
#             _,_,_,_ = self.env.step(self.game_offset)
#             r = -1
#         return [f, r, done]
#
#     def render(self):
#         self.env.render()
#
#     def reset(self):
#         f = self.env.reset()
#         _,_,_,_ = self.env.step(self.game_offset) # force ball spawn
#         self.lives = 5 #info.get('ale.lives')
#         return f
#
#     def preprocess(self, f):
#         f = f[35:-15,:] # cut unnacessary borders
#         f = f[::2, ::2] # resize to  80x80
#         f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
#         _, f = cv2.threshold(f, 1, 255, cv2.THRESH_BINARY) # black white it
#         return f

# neg reward
# class BreakoutWrapper():
#     def __init__(self):
#         self.name = 'Breakout-v0'
#         self.env = gym.make(self.name)
#         self.game_offset = 1
#         self.no_actions = 3
#         self.lives = 5
#         print("valid actions:", self.no_actions, self.env.env.get_action_meanings(),'\n')
#
#     def step(self, action_index):
#         f, r, done, info = self.env.step(action_index + self.game_offset)
#         r = np.clip(r, -1, 1)
#         if self.lives > info.get('ale.lives'):
#             r=-1
#             done = True
#         return [f, r, done]
#
#     def render(self):
#         self.env.render()
#
#     def reset(self):
#         self.lives = 5
#         f = self.env.reset()
#         _,_,_,_ = self.env.step(self.game_offset) # force ball spawn
#         return f
#
#     def preprocess(self, f):
#         f = f[35:-15,:] # cut unnacessary borders
#         f = f[::2, ::2] # resize to  80x80
#         f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
#         _, f = cv2.threshold(f, 1, 255, cv2.THRESH_BINARY) # black white it
#         return f

# no neg rewards
class BreakoutWrapper():
    def __init__(self):
        self.name = 'Breakout-v0'
        self.env = gym.make(self.name)
        self.game_offset = 1
        self.no_actions = 3
        self.lives = 5
        print("valid actions:", self.no_actions, self.env.env.get_action_meanings(),'\n')

    def step(self, action_index):
        f, r, done, info = self.env.step(action_index + self.game_offset)
        r = np.clip(r, 0, 1)
        if self.lives > info.get('ale.lives'):
            done = True
        return [f, r, done]

    def render(self):
        self.env.render()

    def reset(self):
        self.lives = 5
        f = self.env.reset()
        _,_,_,_ = self.env.step(self.game_offset) # force ball spawn
        return f

    def preprocess(self, f):
        f = f[35:-15,:] # cut unnacessary borders
        f = f[::2, ::2] # resize to  80x80
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
        _, f = cv2.threshold(f, 1, 255, cv2.THRESH_BINARY) # black white it
        return f

class PongWrapper():
    def __init__(self):
        self.name = 'Pong-v0'
        self.env = gym.make(self.name)
        self.game_offset = 1
        self.no_actions = 3
        print("valid actions:", self.no_actions, self.env.env.get_action_meanings(),'\n')

    def step(self, action_index):
        f, r, done, info = self.env.step(action_index + self.game_offset)
        r = np.clip(r, -1, 1)
        if r==-1:
            done = True
        return [f, r, done]

    def render(self):
        self.env.render()

    def reset(self):
        self.env.reset()
        for _ in range(19):
            f,_,_,_ = self.env.step(self.game_offset)
        return f

    def preprocess(self, f):
        f = f[35:-15,:] # cut unnacessary borders
        f = f[::2, ::2] # resize to  80x80
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
        _, f = cv2.threshold(f, 100, 255, cv2.THRESH_BINARY ) # black white it
        return f


def separe(x):
    logfile = open("tmp/log.txt","r")
    file = []
    for i in range(x):
        file.append(open("tmp/log_"+str(i)+".txt","a"))
    line =  logfile.readline()
    while line:
        a = int(line[line.find("THREAD")+7])
        print(line[a]+"  "+line)
        file[a].write(line)
        file[a].flush()
        line = logfile.readline()
    for i in range(x):
        file[i].close()


# class CartPoleWrapper():
#     def __init__(self):
#         self.name = 'CartPole-v0'
#         self.env = gym.make(self.name)
#         self.no_actions = 2
#         print("valid actions:", self.no_actions," ['LEFT','RIGHT']\n")
#
#     def step(self, action_index):
#         f, r, done, info = self.env.step(action_index)
#         r = np.clip(r, -1, 1)
#         if done:
#             r = -1
#         return [f, r, done]
#
#     def render(self):
#         self.env.render()
#
#     def reset(self):
#         f = self.env.reset()
#         return f

# class Game():
#     def __init__(self, name):
#         self.name = name
#         self.env = gym.make(self.name)
#         self.game_offset = 1
#         self.no_actions = 3
#         self.lives = 5
#         print("valid actions:", self.no_actions, self.env.env.get_action_meanings(),'\n')
#
#     def step(self, action_index):
#         f, r, done, info = self.env.step(action_index + self.game_offset)
#         lives = info.get('ale.lives')
#
#         if self.lives > lives: # if ball dropped
#             self.lives = lives
#             r = -1
#             if not done:
#                 f,_,_,_ = self.env.step(self.game_offset) # force ball spawn
#
#         r = np.clip(r, -1, 1) # reward clipping
#         return [f, r, done]
#
#     def render(self):
#         self.env.render()
#
#     def reset(self):
#         self.lives = 5
#         self.env.reset()
#         f,_,_,_ = self.env.step(self.game_offset) # force ball spawn
#         return f
