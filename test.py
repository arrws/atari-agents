import gym
from gym import envs
import numpy as np
import random
import imageio
import sys
import threading
# print(envs.registry.all())

from scipy.misc import toimage
import cv2

frame_size = 80
from utils import *

frames = []
frames2 = []
# game = PongWrapper()
game = BreakoutWrapper()

game.reset()
for t in range(1000):
    game.render()

    action = random.randint(0,game.no_actions-1)
    action = 0
    a = input()
    if a!="":
        action = int(a)
    print(action)

    f, reward, done = game.step(action)
    frames.append(f)
    # f = game.preprocess(f)
    # print(sys.getsizeof(frames))
    # toimage(f).show()

    print(reward, done)
    if done:
        print("Episode finished at {} timestep".format(t+1))
        game.reset()

def make_gif(frames):
    frames = np.array(frames)
    imageio.mimsave('t.gif', frames, format='GIF', duration=0.03)
    print(frames.shape)
    print(sys.getsizeof(frames))
