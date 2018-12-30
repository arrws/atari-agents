import numpy as np
import random

from config import *


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
                self.data[k] = 0
            self.data[k] += v

    def log(self, key, val=None, with_min_and_max=False, average_only=False):
        if val is not None:
            print(key,'\t',val)
        else:

            stats = self.get_stats(self.data[key])

            print(key + '\tAvg\t', stats[0])
            if not(average_only):
                print('\tStd\t', stats[1])
            if with_min_and_max:
                print('\tMn/Mx\t', stats[3], '\t', stats[2])
        self.data[key] = []

    def get_stats(self, x):
        mean = np.sum(x) / len(x)
        std = np.sqrt(np.sum(x-mean)**2 / len(x))
        return [mean, std, np.max(x), np.min(x)]



# Buffer Class for storing replay experience

class Buffer():
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.done = []

    def remember_transition(self, t):
        # transition ~ (state, action, reward, result_state, done)
        self.s.append(t[0])
        self.a.append(t[1])
        self.r.append(t[2])
        self.s2.append(t[3])
        self.done.append(t[4])

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

    def get_last_transition(self):
        t = np.dstack(self.s[-5:-1])
        t = np.reshape(t, (-1, config["frame_w"], config["frame_h"], 4))
        return t

    def get_size(self):
        return len(self.s)


