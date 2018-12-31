#!/usr/bin/env python
import numpy as np
import time
import threading

from common import *
from network import *
from wrappers import *
from config import *


"""
Deep Q-Learning with Asynchronous Agents

"""


def run(index, sess, net, target_net, lock, step_counter, env):
    print ("THREAD %d\n" % (index))
    time.sleep(3*index)

    logger = Logger()
    buff = Buffer()

    epsilon = config["init_epsilon"]
    step = 0
    episode = 0

    s = env.reset()
    for i in range(5):
        buff.remember_transition((s, env.get_random_action(), 0, s, False), y=0.1)


    print("\nSTART TRAINNING")
    while step_counter.get() < step_counter.get_max_step():
        step_counter.plus()
        step += 1

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

        y = 0
        if done:
            y = r
        else:
            s_batch = buff.get_last_transition()
            q = sess.run(target_net.q, feed_dict = {target_net.s: s_batch})[0]
            y = r + config["gamma"] * np.max(q)

        buff.remember_transition((s, a, r, s2, done), y)
        s = s2


        # update epsilon
        if epsilon > config['final_epsilon']:
            epsilon -= (config['init_epsilon'] - config['final_epsilon']) / config['anneal_frames']


        # printing and logging
        logger.update( EpSteps = 1,
                       EpScore = r,
                       QvalueAvg = qavg,
                       QvalueMax = qmax,
                       ActionDist = a)


        # TRAIN STEP
        if done or step % config['force_update_freq'] == 0:
            minibatch = buff.get_inorder_minibatch()
            if minibatch:
                s_batch, a_batch, y_batch = minibatch
                sess.run(net.optimizer, feed_dict = { net.a: a_batch,
                                                      net.s: s_batch,
                                                      net.y: y_batch })
                buff.reset(keep_recent=True)


        if done:
            s = env.reset()
            logger.log('THREAD', index)
            logger.log('GlobalStep', step_counter.get())
            logger.log('ThreadStep', step)
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

    print ("THREAD %d has quit\n" % index)


def main():
    start_time = time.time()

    envs = []
    for i in range(config["no_threads"]):
        env = get_environment(config["env_name"])
        envs.append(env)

    lock = threading.Lock()
    step_counter = StepCounter()

    net = Network(env.no_actions)
    target_net = Network(env.no_actions, network=net)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # restore_network(sess, net)
    target_net.copy(sess, net)

    # spawn slave threads
    threads = list()
    for i in range(config["no_threads"]):
        t = threading.Thread(target=run, args=(i, sess, net, target_net, lock, step_counter, envs[i]))
        threads.append(t)

    print("\nStarting Threads...")
    for t in threads:
        t.start()


    last_update = 0
    update_num = 1
    while "I" != "dead":
        now = time.time()
        # for game in envs:
            # game.render()

        if step_counter.get() >= update_num*10*config['target_update_freq'] and now-last_update > 10:
            last_update = now
            update_num += 1
            print ("GLOBAL STEP %d | UPDATING TARGET NETWORK\n" % (step_counter.get()))
            target_net.copy(sess, net)

    # Wait for all of them to finish
    for t in threads:
        t.join()
    sess.close()

main()

