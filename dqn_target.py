#!/usr/bin/env python
import numpy as np
import time

from common import *
from network import *
from wrappers import *
from config import *


"""
Deep Q-Learning with Replay Memory and Target Network

"""


def main():
    start_time = time.time()
    STEP = 0 # global step

    logger = Logger()
    env = get_environment(config["env_name"])
    buff = Buffer()

    net = Network(env.no_actions)
    target_net = Network(env.no_actions, network=net)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # restore_network(sess, net)

    # fill replay memory
    play_random(sess, env, buff)


    def train():
        minibatch = buff.get_minibatch()
        if minibatch:
            [indexes, s_batch, a_batch, r_batch, s2_batch] = minibatch
            q_batch = sess.run(target_net.q, feed_dict = { target_net.s: s2_batch })

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
    episode = 0

    s = env.reset()
    print("\nSTART TRAINNING")
    while True:
        STEP += 1

        # get action
        a, qmax, qavg = [], 0, 0
        if random.random() <= epsilon:
            a = env.get_random_action()
        else:
            s_batch = buff.get_last_transition()
            q = sess.run(net.q, feed_dict = {net.s: s_batch})[0]

            a = np.zeros([env.no_actions])

            a[np.argmax(q)] = 1
            qmax = np.max(q)
            qavg = np.average(q)

        # interact with the environment
        s2, r, done = env.step(a)
        buff.remember_transition((s, a, r, s2, done)) # transition ~ (state, action, reward, result_state)
        s = s2

        # update epsilon
        if epsilon > config['final_epsilon']:
            epsilon -= (config['init_epsilon'] - config['final_epsilon']) / config['anneal_frames']

        if STEP % config['target_update_freq'] == 0:
            target_net.copy(sess, net)

        train()

        # printing and logging
        logger.update( EpSteps = 1,
                       EpScore = r,
                       QvalueAvg = qavg,
                       QvalueMax = qmax,
                       ActionDist = a)

        if done:
            s = env.reset()
            logger.log('GlobalStep', STEP)
            logger.log('Episode', episode)
            logger.log('EpSteps', value_only=True)
            logger.log('EpScore', value_only=True)
            logger.log('QvalueAvg', value_only=True)
            logger.log('QvalueMax', value_only=True)
            logger.log('ActionDist', value_only=True)
            logger.log('Epsilon', epsilon)
            print("")

            episode += 1

            if episode % config['save_freq'] == 0:
                save_gif(buff.get_recent_frames(), "vid_"+str(episode))
                logger.log('Time', time.time()-start_time)

                # save_path = net.saver.save(sess, "tmp/model.ckpt", global_step = step)
                # print("progress logged and model saved in file:", save_path, "\n")

                print("")

main()

