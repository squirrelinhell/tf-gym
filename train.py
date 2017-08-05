
import numpy as np

def train(env, agent, n_episodes = 1000):
    import gym.wrappers
    log_dir = __get_log_dir()
    env = gym.wrappers.Monitor(env, log_dir)

    history = []
    for episode in range(n_episodes):
        obs, reward, done = env.reset(), 0.0, False
        steps, r_sum = 0, 0.0
        while not done:
            action = agent.step(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            steps += 1
            r_sum += reward
        agent.step(obs, reward, done)
        history.append([episode, r_sum, steps])

    env.close()
    __save_plot(log_dir, history)

def __save_plot(log_dir, history):
    history = np.array(history)
    import matplotlib.pyplot as plt
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    __format_plot(ax1, history[:,[0,1]], "reward", "r-")
    __format_plot(ax2, history[:,[0,2]], "length", "b-")
    fig.tight_layout()
    fig.savefig(log_dir + "/by_episode.png")

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

def __get_log_dir():
    import os
    n = 1
    while os.path.exists("__log%05d" % n):
        n += 1
    return "__log%05d" % n
