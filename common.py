import numpy as np
import random
import imageio
import sys

from config import *


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


def save_gif(frames, name='test'):
    imageio.mimsave(name+'.gif', frames, format='GIF', duration=0.03)


# Logger Class for printing

class Logger:
    def __init__(self):
        self.data = dict()

    def store(self, **kwargs):
        for k,v in kwargs.items():
            if not(k in self.data.keys()):
                self.data[k] = []
            self.data[k].append(v)

    def update(self, **kwargs):
        for k,v in kwargs.items():
            if not(k in self.data.keys()):
                self.data[k] = v
            self.data[k] += v

    def log(self, key, val=None, with_min_and_max=False, average_only=False, value_only=False):
        if val is not None:
            print(key,'\t',val)
        else:
            if value_only:
                print(key,'\t', self.data[key])
            else:
                stats = self.get_stats(self.data[key])
                print(key,'\t', stats[0])
                if not(average_only):
                    print('\tStd\t', stats[1])
                if with_min_and_max:
                    print('\tMn/Mx\t', stats[3], '\t', stats[2])
            del self.data[key]

    def get_stats(self, x):
        mean = np.sum(x) / len(x)
        std = np.sqrt(np.sum(x-mean)**2 / len(x))
        return [mean, std, np.max(x), np.min(x)]



# Buffer Class for storing replay experience

class Buffer():
    def __init__(self):
        self.reset()

    def reset(self, keep_recent=False):
        if keep_recent:
            self.s = self.s[-5:]
            self.a = self.a[-5:]
            self.r = self.r[-5:]
            self.s2 = self.s2[-5:]
            self.done = self.done[-5:]
            self.y = self.y[-5:]
            return
        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.done = []
        self.y = []

    def remember_transition(self, t, y=None):
        # transition ~ (state, action, reward, result_state, done)
        self.s.append(t[0])
        self.a.append(t[1])
        self.r.append(t[2])
        self.s2.append(t[3])
        self.done.append(t[4])
        if y:
            self.y.append(y)

        if len(self.s) > config["replay_memory_size"]:
            self.s.pop(0)
            self.a.pop(0)
            self.r.pop(0)
            self.s2.pop(0)
            self.done.pop(0)


    def get_minibatch(self):
        # sample a minibatch to train on
        if len(self.s) > 10:
            indexes = random.sample(range(5, len(self.s)), config["batch_size"])

            s_batch = [np.dstack((self.s[i-3], self.s[i-2], self.s[i-1], self.s[i])) for i in indexes ] # initial states
            a_batch = [self.a[i] for i in indexes] # actions taken
            r_batch = [self.r[i] for i in indexes] # rewards received
            s2_batch = [np.dstack((self.s2[i-3], self.s2[i-2],self.s2[i-1],self.s2[i])) for i in indexes ] # following state
            return [indexes, s_batch, a_batch, r_batch, s2_batch]
        return None

    def get_inorder_minibatch(self):
        if len(self.s) > 10:
            s_batch = [np.dstack((self.s[i-3], self.s[i-2], self.s[i-1], self.s[i])) for i in range(5, self.get_size()) ]
            l = len(s_batch)
            return [s_batch, self.a[-l:], self.y[-l:]]
        return None

    def get_last_transition(self):
        t = np.dstack(self.s[-5:-1])
        t = np.reshape(t, (-1, config["frame_w"], config["frame_h"], 4))
        return t

    def get_size(self):
        return len(self.s)

    def get_recent_frames(self):
        t = np.array(self.s[-100:-1])
        return t


