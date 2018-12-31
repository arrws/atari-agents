import random, numpy, math, gym
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

gamma = 0.99
eta = 0.001
start_epsilon = 1
final_epsilon = 0.01
epsilon = start_epsilon

replay_memory_size = 100000
cumulative_score = 0
update_freq = 1000

episode = -1
step = 0
batch_size = 64

class DQNetwork:
    def __init__(self, input_dim, no_action):
        self.model = self.create_model()
        self.model_ = self.create_model()

    def create_model(self, ):
        model = Sequential()
        model.add(Dense(output_dim=64, activation='relu', input_dim=input_dim))
        model.add(Dense(output_dim=no_actions, activation='linear'))
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def update_target(self):
        self.model_.set_weights(self.model.get_weights())


env = gym.make('CartPole-v0')
input_dim  = env.env.observation_space.shape[0]
no_actions = env.env.action_space.n

nn = DQNetwork(input_dim, no_actions)
D = deque()

def remember_transition(x):
    D.append(x)
    if len(D) > replay_memory_size:
        D.popleft()

def do_something(s):
    a = 0;
    if random.random() < epsilon:
        a = random.randint(0, no_actions-1)
    else:
        a = numpy.argmax(nn.predict(s.reshape(1, input_dim)).flatten())
    s, r, done, info = env.step(a)
    return [s, a, r, done]

def do_train_step():
    batch = random.sample(D, min(batch_size, len(D)))

    s_batch = numpy.array([ o[0] for o in batch ])
    s_batch_ = numpy.array([ o[3] for o in batch ])
    r_batch = numpy.array([ o[2] for o in batch ])
    a_batch = numpy.array([ o[1] for o in batch ])

    p = nn.predict(s_batch)
    p_ = nn.predict(s_batch_, target=True)

    x = numpy.zeros((len(batch), input_dim))
    y = numpy.zeros((len(batch), no_actions))

    for i in range(len(batch)):
        t = p[i]
        if batch[i][4]:
            t[a_batch[i]] = r_batch[i]
        else:
            t[a_batch[i]] = r_batch[i] + gamma*numpy.amax(p_[i])
        y[i] = t
    nn.train(s_batch, y)

while True:
    s = env.reset()
    score = 0
    while True:
        # env.render()
        s_, a, r, done = do_something(s)
        if done:
            s_ = numpy.zeros(input_dim)

        remember_transition((s, a, r, s_, done))

        if step % update_freq == 0:
            nn.update_target()

        epsilon = final_epsilon + (start_epsilon - final_epsilon) * math.exp(-eta*step)
        do_train_step()

        s = s_
        score += r
        step += 1
        if done:
            break

    episode +=1
    cumulative_score += score;
    if episode%100 == 0:
        print("EPISODE:",episode,"SCORE:", cumulative_score/100)
        cumulative_score = 0

