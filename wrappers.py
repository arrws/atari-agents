import numpy as np
import gym
from gym import envs
import cv2
import random


def get_environment(name):
    if name == 'Breakout-v0':
        return BreakoutWrapper()
    elif name == 'CartPole-v0':
        return CartPoleWrapper()
    elif name == 'Pong-v0':
        return PongWrapper()
    return None


class Wrapper():
    def __init__(self, name):
        self.env = gym.make(name)
        self.no_actions = self.env.action_space.n
        print("initialized {}\nactions: {}, {}".format(name, self.no_actions, self.env.env.get_action_meanings()))

    def get_random_action(self):
        a = np.zeros([self.no_actions])
        idx = random.randrange(self.no_actions)
        a[idx] = 1
        return a

    def render(self):
        self.env.render()

    def step(self):
        pass

    def reset(self):
        pass


class BreakoutWrapper(Wrapper):
    def __init__(self):
        super().__init__('Breakout-v0')
        self.game_offset = 1
        self.no_actions = 3
        self.lives = 5

    def step(self, action):
        action_index = np.argmax(action)

        f, r, done, info = self.env.step(action_index + self.game_offset)
        r = np.clip(r, 0, 1)
        if self.lives > info.get('ale.lives'):
            done = True
        f = self.preprocess(f)
        return [f, r, done]

    def reset(self):
        self.lives = 5
        f = self.env.reset()
        _,_,_,_ = self.env.step(self.game_offset) # force ball spawn
        f = self.preprocess(f)
        return f

    def preprocess(self, f):
        f = f[35:-15,:] # cut unnacessary borders
        f = f[::2, ::2] # resize to  80x80
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
        _, f = cv2.threshold(f, 1, 255, cv2.THRESH_BINARY) # black white it
        return f

class PongWrapper():
    def __init__(self):
        super().__init__('Pong-v0')
        self.game_offset = 1
        self.no_actions = 3

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


class CartPoleWrapper():
    def __init__(self):
        self.name = 'CartPole-v0'
        self.env = gym.make(self.name)
        self.no_actions = self.env.action_space.n # for cartpole 2
        print("initialized {}\nvalid actions: {} ['LEFT','RIGHT']\n".format(self.name, self.no_actions))

    def _preprocess_frame(self, f):
        # f = f[35:-15,:] # cut unnacessary borders
        # f = f[::2, ::2] # resize to  80x80
        # f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) # grayscale it
        # _, f = cv2.threshold(f, 1, 255, cv2.THRESH_BINARY) # black white it
        return f

    def step(self, a):
        s, r, done, info = self.env.step(np.argmax(a))
        r = np.clip(r, -1, 1)
        s = self._preprocess_frame(s)
        if done: r = -1 # for cartpole
        return [s, r, done]

    def render(self):
        self.env.render()

    def reset(self):
        s = self.env.reset()
        return [s, False]

    def close(self):
        self.env.close()

