
#Critic (value network) is updated using monte-carlo prediction
#Actor (softmax policy) is updated using TD(0) for Advantage estimation

import gym
import tensorflow as tf
import numpy as np


class Buffer():
    def __init__(self):
        self.reset()

    def reset(self):
        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.ret = []

    def store(self, s, a, r, s2, ret):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s2.append(s2)
        self.ret.append(ret)

    def get_len(self):
        return len(self.s)

    def get_last_episode(self):
        return self.s[-1], self.a[-1], self.r[-1], self.s2[-1], self.ret[-1]



class Actor:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        #Learning parameters
        self.learning_rate = 0.01
        #Declare tf graph
        self.graph = tf.Graph()
        #Build the graph when instantiated
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = tf.Variable(tf.random_normal([len(self.observation_space.high), self.action_space_n]))
            self.biases = tf.Variable(tf.random_normal([self.action_space_n]))

            #Inputs
            self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
            self.y = tf.placeholder("float") #Advantage input
            self.action_input = tf.placeholder("float", [None, self.action_space_n]) #Input action to return the probability associated with that action

            self.policy = tf.nn.softmax(tf.matmul(self.x, self.weights) + self.biases)

            self.log_action_probability = tf.reduce_sum(self.action_input*tf.log(self.policy))
            self.loss = -self.log_action_probability*self.y #Loss is score function times advantage
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #Initializing all variables
            self.init = tf.initialize_all_variables()

        self.sess = tf.Session(graph = self.graph)
        self.sess.run(self.init)


    def rollout_policy(self):
        """Rollout policy for one episode, update the replay memory and return total reward"""
        score = 0
        s = self.env.reset()
        ep_s = []
        ep_a = []
        ep_r = []
        ep_s2 = []
        ep_ret = []

        for time in range(200):
            a = self.choose_action(s)
            s2, r, done, _ = self.env.step(a)
            self.env.render()

            score += r
            if done or time >= self.env.spec.timestep_limit :
                break

            ep_s.append(s)
            ep_a.append(a)
            ep_r.append(r)
            ep_s2.append(s2)
            ep_ret.append(r)
            for i in range(len(ep_ret)-1):
                ep_ret[i] += r

            s = s2

        # for i in range(len(ep_ret)-2, 0, -1):
        #     ep_ret[i] += ep_ret[i+1]

        buff.store(ep_s, ep_a, ep_r, ep_s2, ep_ret)
        return score



    def update_policy(self, advantage_vectors):
        #Update the weights by running gradient descent on graph with loss function defined

        global buff
        for i in range(buff.get_len()):
            states = buff.s[i]
            actions = buff.a[i]

            advantage_vector = advantage_vectors[i]
            for j in range(len(states)):
                action = self.to_action_input(actions[j])

                state = np.asarray(states[j])
                state = state.reshape(1,4)

                _, error_value = self.sess.run([self.optim, self.loss], feed_dict={self.x: state, self.action_input: action, self.y: advantage_vector[j] })


    def choose_action(self, state):
        #Use softmax policy to sample
        state = np.asarray(state)
        state = state.reshape(1,4)
        softmax_out = self.sess.run(self.policy, feed_dict={self.x:state})
        action = np.random.choice([0,1], 1, replace = True, p = softmax_out[0])[0] #Sample action from prob density
        return action




    def to_action_input(self, action):
        action_input = [0]*self.action_space_n
        action_input[action] = 1
        action_input = np.asarray(action_input)
        action_input = action_input.reshape(1, self.action_space_n)
        return action_input




class Critic:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.action_space_n = self.action_space.n
        self.n_input = len(self.observation_space.high)
        self.n_hidden_1 = 20
        #Learning Parameters
        self.learning_rate = 0.008
        # self.learning_rate = 0.1
        self.num_epochs = 20
        self.batch_size = 170
        #Discount factor
        self.discount = 0.90
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1234)
            self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
            }
            self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([1]))
            }
            self.s = self.x = tf.placeholder("float", [None, len(self.observation_space.high)])#State input
            self.y = tf.placeholder("float") #Target return

            layer_1 = tf.add(tf.matmul(self.s, self.weights['h1']), self.biases['b1'])
            layer_1 = tf.nn.tanh(layer_1)
            self.value_pred = tf.matmul(layer_1, self.weights['out']) + self.biases['out']


            self.loss = tf.reduce_mean(tf.pow(self.value_pred - self.y,2))
            self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            init = tf.initialize_all_variables()

        self.sess = tf.Session(graph = self.graph)
        self.sess.run(init)


    def update_value_estimate(self):
        global buff
        #Monte Carlo prediction
        batch_size = min(buff.get_len(), self.batch_size)
        for epoch in range(self.num_epochs):
            #Loop over all batches
            for i in range( buff.get_len()//batch_size ):
                batch_s, batch_y = self.get_next_batch(batch_size, buff.s, buff.ret)

                #Fit training data using batch
                self.sess.run(self.optim, feed_dict={self.s:batch_s, self.y:batch_y})


    def get_advantage_vector(self, states, rewards, next_states):
        #Return TD(0) Advantage for particular state and action
        #Get value of current state
        advantage_vector = []
        for i in range(len(states)):
            state = np.asarray(states[i])
            state = state.reshape(1,4)
            next_state = np.asarray(next_states[i])
            next_state = next_state.reshape(1,4)
            reward = rewards[i]
            state_value = self.sess.run(self.value_pred, feed_dict={self.s:state})
            next_state_value = self.sess.run(self.value_pred, feed_dict={self.s:next_state})
            #Current implementation uses TD(0) advantage
            advantage = reward + self.discount*next_state_value - state_value
            advantage_vector.append(advantage)

        return advantage_vector


    def get_next_batch(self, batch_size, states, returns):
        #Return mini-batch of transitions from replay data
        s = []
        r = []
        for i in range(len(states)):
            for j in range(len(states[i])):
                s.append(states[i][j])
                r.append(returns[i][j])
        s = np.asarray(s)
        r = np.asarray(r)
        idx =  np.random.randint(s.shape[0], size=batch_size)
        return s[idx, :], r[idx]



buff = Buffer()
env = gym.make('CartPole-v0')
env.seed(1234)
np.random.seed(1234)
#env.monitor.start('./cartpole-pg-experiment-15')
#Learning Parameters
max_episodes = 1000
episodes_before_update = 2

actor = Actor(env)
critic = Critic(env)



def run():

    advantage_vectors = []
    sum_reward = 0
    update = True
    for i in range(max_episodes):

        episode_score = actor.rollout_policy()
        episode_states, episode_actions, episode_rewards, episode_next_states, episode_return_from_states = buff.get_last_episode()
        advantage_vector = critic.get_advantage_vector(episode_states, episode_rewards, episode_next_states)
        advantage_vectors.append(advantage_vector)
        sum_reward += episode_score

        if (i+1)%episodes_before_update == 0:
            avg_reward = sum_reward/episodes_before_update
            print("Current {} episode average reward: {}".format(i, avg_reward))

            if avg_reward >= 195:
                print("Passed")
            else:
                actor.update_policy(advantage_vectors)
                critic.update_value_estimate()

            #Delete the data collected so far
            del advantage_vectors[:]
            buff.reset()
            sum_reward = 0

run()

