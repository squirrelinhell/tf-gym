#!/usr/bin/env python3

import os
import sys
import numpy as np

import debug

def train(env, agent, n_steps, results_file = None):
    results_file = __check_output_path(results_file, "RESULTS_FILE")
    history = []
    obs, reward, done = env.reset(), 0.0, False
    ep_r, ep_steps = 0.0, 0

    for t in range(n_steps):
        action = agent.step(obs, reward, done)

        if done:
            history.append([t + 1, len(history) + 1, ep_r, ep_steps])
            obs, reward, done = env.reset(), 0.0, False
            ep_r, ep_steps = 0.0, 0
        else:
            obs, reward, done, _ = env.step(action)
            ep_r += reward
            ep_steps += 1


    if results_file is not None:
        np.savetxt(
            results_file,
            history,
            fmt="%9d %9d %9.2f %9d",
            header="   step   episode ep_reward  ep_steps"
        )

def plot_results(*result_files):
    plot_file = __check_output_path(None, "PLOT_FILE")

    import matplotlib
    if plot_file is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    ax1.set_ylabel("reward/episode")
    ax2.set_ylabel("steps/episode")
    ax2.set_xlabel("step")
    patches = []

    for f, color in zip(result_files, ["r", "b", "g", "c", "m"]):
        data = np.genfromtxt(f, names=True)
        __add_plot(ax1, data["step"], data["ep_reward"], color)
        __add_plot(ax2, data["step"], data["ep_steps"], color)
        patches.append(matplotlib.patches.Patch(color=color))

    if len(result_files) >= 2:
        n = [os.path.basename(f).split(".")[0] for f in result_files]
        plt.figlegend(handles=patches, labels=n, loc=3)

    if plot_file is None:
        plt.show()
    else:
        fig.set_size_inches(8, 6)
        fig.savefig(plot_file, dpi=100)

def __batch_avg(data, batch):
    n = len(data) // batch
    data = data[0:batch*n]
    data = data.reshape((n, batch) + data.shape[1:])
    return np.mean(data, axis=1)

def __add_plot(ax, x, y, color):
    xy = np.vstack((x, y)).T
    ax.plot(xy[:,0], xy[:,1], color + ".", alpha=0.2, zorder=1)
    if len(xy) >= 20:
        xy = __batch_avg(xy, max(5, len(xy) // 100))
        ax.plot(xy[:,0], xy[:,1], color + "-", zorder=2)
    ax.grid()
    ax.figure.tight_layout()

def __check_output_path(path, env=None):
    if path is None and env is not None:
        if env in os.environ and len(os.environ[env]) >= 1:
            path = os.environ[env]
    if path is not None:
        if os.path.exists(path):
            raise ValueError("File already exists: %s" % path)
    return path

def run():
    if len(sys.argv) >= 3 and sys.argv[1] == "plot":
        plot_results(*sys.argv[2:])
    else:
        sys.stderr.write("\nUsage:\n\n")
        sys.stderr.write("\t" + "train.py plot <results.txt>\n")
        sys.stderr.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    run()
