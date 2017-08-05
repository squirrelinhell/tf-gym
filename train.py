
import os
import numpy as np

def train(env, agent, n_steps, log_dir = None):
    if log_dir is None:
        log_dir = __get_log_dir()
    else:
        if os.path.exists(log_dir):
            raise ValueError("Directory already exists: %s" % log_dir)

    import gym.wrappers
    env = gym.wrappers.Monitor(env, log_dir)

    done = True
    history = []
    r_sum, steps = 0.0, 0
    for t in range(n_steps):
        if done:
            history.append([t, len(history), r_sum, steps])
            r_sum, steps = 0.0, 0
            obs, reward, done = env.reset(), 0.0, False
        else:
            obs, reward, done, _ = env.step(action)
            r_sum += reward
            steps += 1
        action = agent.step(obs, reward, done)

    env.close()
    __save_plot(log_dir, history)

def __save_plot(log_dir, history):
    history = np.array(history)
    import matplotlib.pyplot as plt

    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    ax2.set_xlabel("episode")
    __format_plot(ax1, history[:,[1,2]], "reward", "r-")
    __format_plot(ax2, history[:,[1,3]], "episode length", "b-")
    fig.savefig(log_dir + "/by_episode.png")

    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    ax2.set_xlabel("step")
    __format_plot(ax1, history[:,[0,2]], "reward", "r-")
    __format_plot(ax2, history[:,[0,3]], "episode length", "b-")
    fig.savefig(log_dir + "/by_step.png")

def __batch_avg(data, batch):
    n = len(data) // batch
    data = data[0:batch*n]
    data = data.reshape((n, batch) + data.shape[1:])
    return np.mean(data, axis=1)

def __format_plot(ax, xy, label, style):
    if len(xy) > 100:
        ax.plot(xy[:,0], xy[:,1], "y.")
        xy = __batch_avg(xy, len(xy) // 100)
    if len(xy) > 20:
        ax.plot(xy[:,0], xy[:,1], "k.")
        xy = __batch_avg(xy, len(xy) // 20)
    ax.plot(xy[:,0], xy[:,1], style)
    ax.set_ylabel(label, color=style[0])
    ax.tick_params("y", colors=style[0])
    ax.grid()
    ax.figure.tight_layout()

def __get_log_dir():
    n = 1
    while os.path.exists("/tmp/log%05d" % n):
        n += 1
    return "/tmp/log%05d" % n
