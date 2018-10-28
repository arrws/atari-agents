#!/usr/bin/env python
# BILLY
import threading
import tensorflow as tf
import cv2
import sys
import numpy as np
import random
import gym
from gym import envs
import time

from utils import *

# PARAMS
no_threads = 4 # number of learning agents
frame_size = 80 # frames will be preprocessed to 80x80 images
update_model_freq = 32 # how often to train model ~ batch_size
update_target_freq = 10000 # how often to update target network

learn_rate = 0.00025 # learning rate
gamma = 0.99 # reward decay rate of past observations

start_epsilon = 1. # starting value of epsilon
anneal_frames = 1000000. # frames over which to anneal epsilon

# LOGGING
log_freq = 100 # after how many episodes to log and save network
logfile = open("tmp/log.txt","a")
start_time = time.time()
current_time = time.time()


def restore_network(sess, saver):
    # load network if exists
    checkpoint = tf.train.get_checkpoint_state("tmp")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully restored:", checkpoint.model_checkpoint_path)
    else:
        print("Could not restore network")

def sample_final_epsilon():
    final_epsilons = np.array([.1,.01,.2])
    probabilities = np.array([0.4,0.3,0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]


def run_agent(index, sess, agent_nn, target_nn, saver, lock, step_counter, game):
    epsilon = start_epsilon
    final_epsilon = sample_final_epsilon()
    # epsilon = final_epsilon

    # intialize starting s
    f = game.preprocess(game.reset())
    s = np.stack((f, f, f, f), axis = 2)

    s_batch = []
    y_batch = []
    a_batch = []
    step = 0.0

    ep_score = 0.0
    ep_qmax = 0.0
    ep_steps = 0.0
    episodes = 0
    cumulative_score = 0.
    cumulative_steps = 0.
    cumulative_qmax = 0.

    print ("Starting THREAD %d with FINAL EPSILON %0.5f \n" % (index, final_epsilon))
    time.sleep(3*index)


    while step_counter.get() < step_counter.get_max_step():
        step_counter.plus()
        step += 1.0
        ep_steps += 1.0

        # anneal explorativeness
        if epsilon > final_epsilon:
            epsilon -= (start_epsilon - final_epsilon) / anneal_frames # update epsilon

        # save new observation to s ( a s consists of 4 consecutive observations )
        s_batch.append(s)

        # evaluate all action values at current s
        action_index = 0
        if random.random() < epsilon:
            action_index = random.randrange(game.no_actions)
        else:
            q = agent_nn.evaluate(sess, [s])
            action_index = np.argmax(q)

        a = np.zeros([game.no_actions])
        a[action_index] = 1
        a_batch.append(a)

        f, r, done = game.step(action_index)
        f = game.preprocess(f)

        ep_score += r
        if done:
            y_batch.append(r)
        else:
            # evaluate the loss with the target network
            s = np.append(np.reshape(f, (frame_size, frame_size, 1)), s[:, :, :3], axis=2)
            qmax = np.max(target_nn.evaluate(sess, [s]))
            ep_qmax += qmax
            y_batch.append(r + gamma * qmax)

        if done or (step % update_model_freq == 0):
            agent_nn.train(sess, a_batch, s_batch, y_batch)
            s_batch = []
            y_batch = []
            a_batch = []

        if done:
            f = game.preprocess(game.reset())
            s = np.stack((f, f, f, f), axis = 2)

            print("GLOBAL STEP %d | THREAD %d | STEP %d | EPISODE %d | SCORE %d | LENGTH %d | Q %.5f | EPSILON %.5f" % (step_counter.get(), index, step, episodes, ep_score, ep_steps, ep_qmax/ep_steps, epsilon))
            cumulative_score += ep_score
            cumulative_steps += ep_steps
            cumulative_qmax += ep_qmax
            ep_steps = 0.0
            ep_score = 0.0
            ep_qmax = 0.0
            episodes += 1

            if episodes % log_freq == 0:
                current_time = time.time()
                logfile.write("GLOBAL STEP %d | THREAD %d | STEP %d | EPISODE %d | AVG SCORE %.2f | AVG LENGTH %.2f | AVG Q %.2f | EPSILON %.5f | TIME PASSED %d\n" % (step_counter.get(), index, step, episodes, cumulative_score/log_freq, cumulative_steps/log_freq, cumulative_qmax/log_freq, epsilon, int(current_time - start_time)))
                logfile.flush()

                if index == no_threads-1:
                    save_path = saver.save(sess, "tmp/model.ckpt", global_step = int(step))
                    print("progress logged and model saved in file:", save_path, "\n")

                cumulative_score = 0.
                cumulative_steps = 0.
                cumulative_qmax = 0.

    print ("THREAD %d has quit\n" % index)


def main():
    # initialize GAME environments
    envs = []
    for i in range(no_threads):
        game = BreakoutWrapper()
        envs.append(game)

    lock = threading.Lock()
    step_counter = StepCounter()

    agent_nn = DQNetwork(frame_size, envs[0].no_actions, learn_rate, None)
    target_nn = DQNetwork(frame_size, envs[0].no_actions, learn_rate, agent_nn)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver(max_to_keep = None)
    restore_network(sess, saver)
    target_nn.copy(sess, agent_nn)

    # spawn agent threads
    threads = list()
    for i in range(no_threads):
        t = threading.Thread(target=run_agent, args=(i, sess, agent_nn, target_nn, saver, lock, step_counter, envs[i]))
        threads.append(t)

    # Start all threads
    print("\nStarting Threads...")
    for t in threads:
        t.start()

    last_update = 0
    update_num = 1
    while "I" != "dead":
        now = time.time()
        for game in envs:
            game.render()
        if step_counter.get() >= update_num*update_target_freq and now-last_update > 10:
            last_update = now
            update_num += 1
            print ("GLOBAL STEP %d | UPDATING TARGET NETWORK\n" % (step_counter.get()))
            target_nn.copy(sess, agent_nn)

    # Wait for all of them to finish
    for t in threads:
        t.join()

    sess.close()

main()
