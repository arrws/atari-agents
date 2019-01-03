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


def run(index, sess, trainer, lock, STEP, env):
    print ("THREAD %d" % (index))
    time.sleep(3*index)

    name = "worker_"+str(index)
    local_net = AC_Network(env.no_actions, name, trainer)
    update_local_ops = update_target_graph('global', name)

    logger = Logger()
    buff = Buffer()

    epsilon = config["init_epsilon"]
    step = 0
    episode = 0

    s = env.reset()
    for i in range(5): # dummy fill
        buff.remember_transition((s, env.get_random_action(), 0, s, False))


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
            p = sess.run(local_net.policy, feed_dict = {local_net.s: s_batch})[0]
            idx = np.random.choice(env.no_actions, p=p)
            a = np.zeros([env.no_actions])
            a[idx] = 1

            logger.update( EpPolicy = p )


        # interact with the environment
        s2, r, done = env.step(a)
        buff.remember_transition((s, a, r, s2, done))
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
                s_batch, a_batch, r_batch = minibatch

                v = r
                if not done: # bootstrap
                    v = sess.run(local_net.value, feed_dict = {local_net.s: s_batch})[0]

                discounted_r_batch = discount(r_batch)
                v_batch = np.asarray(v_batch.tolist() + [v])
                adv_batch = r_batch - v_batch
                discounted_adv_batch = discount(adv_batch)

                result = sess.run([ net.value_loss,
                                    net.policy_loss,
                                    net.entropy,
                                    net.grad_norms,
                                    net.var_norms,
                                    net.state_out,
                                    net.apply_grads],
                                    feed_dict = {   net.y: discounted_r_batch,
                                                    net.s: s_batch,
                                                    net.a: a_batch,
                                                    net.advantages: discounted_adv_batch}
                                  )
                buff.reset(keep_recent=True)
                sess.run(self.update_local_ops)


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

                # save_path = local_net.saver.save(sess, "tmp/model.ckpt", global_step = step)
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


    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    target_net = AC_Network(env.no_actions, 'global', None)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # restore_network(sess, net)


    # spawn slave threads
    print("\nSpawning Threads...")
    lock = threading.Lock()
    threads = list()
    for i in range(config["no_threads"]):
        t = threading.Thread(target=run, args=(i, sess, trainer, lock, STEP, envs[i]))
        threads.append(t)
        threads[-1].start()

    # Wait for all threads to finish
    for t in threads:
        t.join()
    sess.close()

main()

