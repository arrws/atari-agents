#!/usr/bin/env python
import numpy as np
import time

from utils import *
from network import *
from wrappers import *
from config import *


def main():
    start_time = time.time()
    STEP = 0 # global step

    logger = Logger()
    env = get_environment(config["env_name"])
    buff = Buffer()

    net = Network(env.no_actions)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # restore_network(sess, net)

    def play_random(sess):
        s = env.reset()
        for step in range(config["observe_frames"]): # to fill the replay memory
            a = env.get_random_action()
            s2, r, done = env.step(a)
            buff.remember_transition((s, a, r, s2, done))
            s = s2
            if step % 1000 == 0:
                print("STEP %d" % (step))
            if done:
                s = env.reset()

    print("\nFill Replay Memory")
    play_random(sess) # fill replay memory



    def train():
        minibatch = buff.get_minibatch()
        if minibatch:
            [indexes, s_batch, a_batch, r_batch, s2_batch] = minibatch
            q_batch = sess.run(net.q, feed_dict = {net.s: s2_batch})

            y_batch = []
            for i, index in enumerate(indexes):
                # if episode is done
                if buff.done[index]:
                    # last reward
                    y_batch.append(r_batch[i])
                else:
                    # predicted reward plus discounted prediction
                    y_batch.append(r_batch[i] + config["gamma"] * np.max(q_batch[i]))

            # perform gradient step
            sess.run(net.optimizer, feed_dict = { net.a: a_batch,
                                                net.s: s_batch,
                                                net.y: y_batch })



    epsilon = config["init_epsilon"]
    score = 0.
    steps = 0.
    episodes = 0

    s = env.reset()
    print("\nSTART TRAINNING")
    while True:

        # get action
        a = []
        if random.random() <= epsilon:
            a = env.get_random_action()
        else:
            s_batch = buff.get_last_transition()
            q = sess.run(net.q, feed_dict = {net.s: s_batch})[0]

            a = np.zeros([env.no_actions])
            a[np.argmax(q)] = 1


        s2, r, done = env.step(a)

        buff.remember_transition((s, a, r, s2, done)) # transition ~ (state, action, reward, result_state)
        s = s2

        # update epsilon
        if epsilon > config['final_epsilon']:
            epsilon -= (config['init_epsilon'] - config['final_epsilon']) / config['anneal_frames']


        train()

        STEP += 1
        steps += 1
        score += r

        if done:
            s = env.reset()
            print("STEP %d | EPISODE %d | SCORE %d | LENGTH %d | | EPSILON %.5f" % (STEP, episodes, score, steps, epsilon))

            logger.store(
                      EpSteps = steps,
                      EpScore = score,
                      )

            steps = 0.
            score = 0.
            episodes += 1

            # logfile = open("tmp/log.txt","a")
            if episodes % 10 == 0:
                logger.log('EpSteps', average_only=True)
                logger.log('EpScore', with_min_and_max=True)
                logger.log('Time', time.time()-start_time)
                print("")

                # logfile.write("STEP %d | EPISODE %d | AVG SCORE %.2f | AVG LENGTH %.2f | AVG Q %.2f | EPSILON %.5f | TIME PASSED %d\n" % (step, episodes, cumulative_score/log_freq, cumulative_steps/log_freq, cumulative_qmax/log_freq, epsilon, int(current_time - start_time)))
                # logfile.flush()

                # save_path = net.saver.save(sess, "tmp/model.ckpt", global_step = step)
                # print("progress logged and model saved in file:", save_path, "\n")

    # logfile.close()

main()

