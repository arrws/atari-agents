config = \
{
    "name" : "dqn",
    "logfile" : "tmp/log.txt",

    "env_name" : "Breakout-v0",

    "batch_size" : 1,
    "frame_w" : 80,
    "frame_h" : 80,

    "learn_rate" : 0.00025,
    "gamma" : 0.99,


    "init_epsilon" : 1.,
    "final_epsilon" : 0.1,

    "observe_frames" : 100,
    "anneal_frames" : 50,
    "replay_memory_size" : 100,

    "save_freq" : 2,

    "target_update_freq" : 30,
    "force_update_freq" : 50,
    "no_threads" : 2,

    "loss_v" : 0.5,
    "loss_entropy" : 0.01,

}
