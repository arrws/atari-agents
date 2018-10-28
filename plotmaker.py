import matplotlib.pyplot as plt
import numpy as np

xname = "STEP"
# xname = "EPISODE"
yname = "AVG SCORE"
# yname = "AVG LENGTH"
# yname = "AVG Q"
plt.xlabel(xname)
plt.ylabel(yname)
plt.grid(True)
plt.title('Performance Evaluation')

def get_float(line,s):
    x = line.find(s)+len(s)+1
    y = line[x:].find("|")
    # print(line[x:x+y])
    # print(float(line[x:x+y]))
    return float(line[x:x+y])

def plot_logfile(filename, name):
    logfile = open(filename,"r")
    x = []
    y = []
    for line in logfile.readlines():
        x.append(get_float(line, xname))
        y.append(get_float(line, yname))
    plt.plot(x, y, label=name)

# for neg rewards add 1
def plot_logfile2(filename, name):
    logfile = open(filename,"r")
    x = []
    y = []
    for line in logfile.readlines():
        x.append(get_float(line, xname))
        y.append(get_float(line, yname)+1)
    plt.plot(x, y, label=name)


# for async has no_threads
def plot_logfile_threaded(filename, label):
    logfile = open(filename,"r")
    no_threads = 6
    x = [[] for i in range(no_threads)]
    y = [[] for i in range(no_threads)]

    lines = logfile.readlines()
    k = lines.index("x\n")

    for line in lines[:k]:
        t = int(get_float(line, "THREAD"))
        x[t].append(get_float(line, "| STEP"))
        y[t].append(get_float(line, yname))

    z = [len(x[t])-1 for t in range(no_threads)]

    for line in lines[k+1:]:
        t = int(get_float(line, "THREAD"))
        x[t].append(get_float(line, "| STEP")+x[t][z[t]])
        y[t].append(get_float(line, yname))

    k = min(len(x[i]) for i in range(no_threads))
    avg_x = [(sum(x[i][j] for i in range(no_threads))/no_threads) for j in range(k)]
    avg_y = [(sum(y[i][j] for i in range(no_threads))/no_threads) for j in range(k)]

    # for i in [0,2,3]:
    #     plt.plot(x[i], y[i], label= "thread "+str(i))
    plt.plot(avg_x, avg_y, label=label)

def main():
    # plot_logfile2("tmp_ddqn_bignn_adam_negreward/log.txt", ">>>")
    # plot_logfile2("tmp_ddqn_bignn_rmsprop_negreward/log.txt", "negative rewards")
    # plot_logfile("tmp_ddqn_bignn_rmsprop_posreward/log.txt", "improved Big NN")
    # plot_logfile("tmp_ddqn_bignn_rmsprop_posreward/log_bigeps.txt", "big")
    # plot_logfile("tmp_ddqn_bignn_rmsprop_posreward/log_smalleps.txt", "small")
    # plot_logfile2("tmp_ddqn_smallnn_rmsprop_negreward/log.txt", "all")
    plot_logfile_threaded("tmp_adqn_bignn_rmsprop_posreward/log.txt", "async Big NN")
    plot_logfile_threaded("tmp_adqn_smallnn_adam_posreward/log.txt", "async Small NN")
    # plot_logfile("tmp_dqn_smallnn_adam_posrewards/log.txt","simple Small NN")
    # plt.savefig("test.png")
    plt.legend()
    plt.show()

main()
