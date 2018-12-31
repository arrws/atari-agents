#!/usr/bin/env python
import numpy as np
import time
import threading

from common import *
from network import *
from wrappers import *
from config import *


"""
Actor Critic with asynchronous updates

"""


def run(index, sess, net, target_net, lock, STEP, env):
    print ("THREAD %d" % (index))
    time.sleep(3*index)

    logger = Logger()
    buff = Buffer()

    epsilon = config["init_epsilon"]
    step = 0
    episode = 0

    s = env.reset()
    for i in range(5): # dummy fill
        buff.remember_transition((s, env.get_random_action(), 0, s, False), y=0.1)


    # trainning
    while STEP.get() < STEP.get_max_step():
        STEP.plus()
        step += 1

        # get action
        a, qmax, qavg = [], 0, 0
        if random.random() <= epsilon:
            a = env.get_random_action()
        else:
            s_batch = buff.get_last_transition()
            p = sess.run(net.policy, feed_dict = {net.s: s_batch})[0]
            idx = np.random.choice(env.no_actions, p=p)
            a = np.zeros([env.no_actions])
            a[idx] = 1

            logger.update( EpPolicy = p )


        # interact with the environment
        s2, r, done = env.step(a)

        # compute target return
        y = 0
        if done:
            y = r
        else:
            s_batch = buff.get_last_transition()
            v = sess.run(target_net.value, feed_dict = {target_net.s: s_batch})[0]
            y = r + config["gamma"] * v[0]

            logger.update( EpValue = v )

        buff.remember_transition((s, a, r, s2, done), y)
        s = s2


        # update epsilon
        if epsilon > config['final_epsilon']:
            epsilon -= (config['init_epsilon'] - config['final_epsilon']) / config['anneal_frames']


        logger.update( EpSteps = 1,
                       EpScore = r,
                       ActionDist = a)


        # NETWORK TRAIN STEP
        if done or step % config['force_update_freq'] == 0:
            minibatch = buff.get_inorder_minibatch()
            if minibatch:
                s_batch, a_batch, y_batch = minibatch
                sess.run(net.optimizer, feed_dict = { net.a: a_batch,
                                                      net.s: s_batch,
                                                      net.y: y_batch })
                buff.reset(keep_recent=True)


        # printing and logging
        if done:
            s = env.reset()

            data = logger.get_data(['EpSteps', 'EpScore', 'EpPolicy', 'EpValue', 'ActionDist'])
            print("THREAD "+str(index)+" results:")
            logger.log('GlobalStep', STEP.get())
            logger.log('ThreadStep', step)
            logger.log('Episode', episode)
            logger.log('EpSteps', value_only=True)
            logger.log('EpScore', value_only=True)
            logger.log('EpPolicy', value_only=True)
            logger.log('EpValue', value_only=True)
            logger.log('ActionDist', value_only=True)
            logger.log('Epsilon', epsilon)
            print("")

            logger.store( CumSteps = data['EpSteps'],
                          CumScore = data['EpScore'],
                          CumPolicy = data['EpPolicy'],
                          CumValue = data['EpValue'],
                          CumActions = data['ActionDist'],
                         )

            episode += 1

            if episode % config['save_freq'] == 0:
                print("THREAD "+str(index)+" STATS per "+str(config['save_freq'])+" episodes:")
                logger.log('CumSteps', with_min_and_max=True)
                logger.log('CumScore', with_min_and_max=True)
                logger.log('CumPolicy', average_only=True)
                logger.log('CumValue', average_only=True)
                logger.log('CumActions', average_only=True)

                # save_gif(buff.get_recent_frames(), "vid_"+str(index)+"_"+str(episode))

                # save_path = net.saver.save(sess, "tmp/model.ckpt", global_step = step)
                # print("progress logged and model saved in file:", save_path, "\n")

                print("")

    print ("THREAD %d has quit" % index)


def main():
    start_time = time.time()
    STEP = StepCounter()

    envs = []
    for i in range(config["no_threads"]):
        env = get_environment(config["env_name"])
        envs.append(env)


    net = AC_Network(env.no_actions)
    target_net = AC_Network(env.no_actions, network=net)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # restore_network(sess, net)
    target_net.copy(sess, net)


    # spawn slave threads
    print("\nSpawning Threads...")
    lock = threading.Lock()
    threads = list()
    for i in range(config["no_threads"]):
        t = threading.Thread(target=run, args=(i, sess, net, target_net, lock, STEP, envs[i]))
        threads.append(t)
        threads[-1].start()

    prev = 0
    updates = 0
    step_target_update = 20*config['target_update_freq']

    while True:
        now = time.time()

        # for env in envs:
        #     env.render()

        if STEP.get() >= updates*step_target_update  and now-prev > 10:
            prev = now
            updates += 1
            target_net.copy(sess, net)
            print ("GLOBAL STEP %d updated target network" % (STEP.get()))

    # Wait for all threads to finish
    for t in threads:
        t.join()
    sess.close()

main()

