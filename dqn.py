#!/usr/bin/env python
# JOHNNY
import tensorflow as tf
import numpy as np
import gym
from gym import envs
from collections import deque
import cv2
import sys
import random
import time
import os
import imageio

from utils import *

# PARAMS
batch_size = 32 # size of minibatch
frame_size = 80 # frames will be preprocessed to 80x80 images

learn_rate = 0.00025 # learning rate
gamma = 0.99 # reward decay rate of past observations

start_epsilon = 1. # starting value of epsilon
final_epsilon = 0.1 # final value of epsilon
epsilon = start_epsilon # current epsilon
anneal_frames = 1000000 # frames over which to anneal epsilon

replay_memory_size = 400000 # number of previous transitions to remember
observe_frames = 50000 # timesteps to choose actions randomly play
# observe_frames = 50

# LOGGING
log_freq = 100 # after how many episodes to log and save network
logfile = open("tmp/log.txt","a")
start_time = time.time()
current_time = time.time()

# INIT
step = 0 # global step
game = BreakoutWrapper()
D = deque() # replay memory
nn = DQNetwork(frame_size, game.no_actions, learn_rate, None)


def remember_transition(transition): # transition ~ (state, action, reward, result_state)
    global D
    D.append(transition)
    if len(D) > replay_memory_size:
        D.popleft()

def restore_network(sess):
    # load network if exists
    global step
    checkpoint = tf.train.get_checkpoint_state("tmp")
    if checkpoint and checkpoint.model_checkpoint_path:
        nn.saver.restore(sess, checkpoint.model_checkpoint_path)
        step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
        print("Successfully restored:", checkpoint.model_checkpoint_path,"\nglobal step =",step)
    else:
        print("Could not restore network\nglobal step =",step)

def do_train_step(sess):
    global D
    # sample a minibatch to train on
    indexes = random.sample(range(5, len(D)), batch_size)
    s_batch = [np.dstack((D[i-3][0], D[i-2][0],D[i-1][0],D[i][0])) for i in indexes ] # initial states
    a_batch = [D[i][1] for i in indexes] # actions taken
    r_batch = [D[i][2] for i in indexes] # rewards received
    s2_batch = [np.dstack((D[i-3][3], D[i-2][3],D[i-1][3],D[i][3])) for i in indexes ] # following state

    y_batch = []
    q_batch = nn.evaluate(sess, s2_batch)

    for i, index in enumerate(indexes):
        if D[index][4]: # if episode is done
            y_batch.append(r_batch[i]) # last reward
        else:
            y_batch.append(r_batch[i] + gamma * np.max(q_batch[i])) # predicted reward plus discounted prediction

    nn.train(sess, a_batch, s_batch, y_batch) # perform gradient step

def do_something(sess, s):
    global D
    global env
    global step
    action_index = 0

    if (random.random() <= epsilon or step <= observe_frames) or len(D)<4: # !!! delete this for train
        action_index = random.randrange(game.no_actions)
        qmax = 0
    else:
        s_batch = np.reshape(np.dstack([D[-3][0], D[-2][0], D[-1][0], s]), (-1, frame_size, frame_size, 4))
        q = nn.evaluate(sess, s_batch)[0] # q is in double [[ ]]
        action_index = np.argmax(q)
        qmax = np.max(q)

    a = np.zeros([game.no_actions])
    a[action_index] = 1

    # game.render()
    f, r, done = game.step(action_index)
    f = game.preprocess(f)
    return [f, qmax, a, r, done]

def play_random(sess):
    global env
    global step

    s = game.preprocess(game.reset())
    print("\nSTART OBSERVING")

    for obs_step in range(0, observe_frames): # to fill the replay memory
        s2, qmax, a, r, done = do_something(sess, s)
        remember_transition((s, a, r, s2, done))
        s = s2
        if obs_step % 5000 == 0:
            print("STEP %d | EPSILON %.5f" % (obs_step, epsilon))
        if done:
            s = game.preprocess(game.reset())
    if step == 0: # if we didnt load any other network to continue trainning
        step = observe_frames

def main():
    global D
    global step
    global epsilon
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    restore_network(sess)
    play_random(sess) # fill replay memory

    ep_score = 0.
    ep_qmax = 0.
    ep_steps = 0.
    episodes = 0
    cumulative_score = 0.
    cumulative_steps = 0.
    cumulative_qmax = 0.

    s = game.preprocess(game.reset())
    print("\nSTART TRAINNING")

    while True:
        s2, qmax, a, r, done = do_something(sess, s)
        remember_transition((s, a, r, s2, done)) # transition ~ (state, action, reward, result_state)
        s = s2

        if epsilon > final_epsilon: # anneal explorativeness
            epsilon -= (start_epsilon - final_epsilon) / anneal_frames # update epsilon

        do_train_step(sess)

        step += 1
        ep_steps += 1
        ep_score += r
        ep_qmax += qmax

        # saving printing logging ...
        # print("STEP %d | EPISODE %d | EPSILON %.5f | ACTION %d | QMAX %.5f | REWARD %.5f" % (step, episodes, epsilon, np.argmax(a), qmax, r))

        if done:
            s = game.preprocess(game.reset())

            print("STEP %d | EPISODE %d | SCORE %d | LENGTH %d | Q %.5f | EPSILON %.5f" % (step, episodes, ep_score, ep_steps, ep_qmax, epsilon))
            cumulative_score += ep_score
            cumulative_steps += ep_steps
            cumulative_qmax += ep_qmax
            ep_steps = 0.
            ep_score = 0.
            ep_qmax = 0.
            episodes += 1

            if episodes % log_freq == 0:
                current_time = time.time()
                logfile.write("STEP %d | EPISODE %d | AVG SCORE %.2f | AVG LENGTH %.2f | AVG Q %.2f | EPSILON %.5f | TIME PASSED %d\n" % (step, episodes, cumulative_score/log_freq, cumulative_steps/log_freq, cumulative_qmax/log_freq, epsilon, int(current_time - start_time)))
                logfile.flush()

                save_path = nn.saver.save(sess, "tmp/model.ckpt", global_step = step)
                print("progress logged and model saved in file:", save_path, "\n")

                cumulative_score = 0.
                cumulative_steps = 0.
                cumulative_qmax = 0.

    logfile.close()

main()
