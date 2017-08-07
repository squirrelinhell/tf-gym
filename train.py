#!/usr/bin/env python3

import os
import sys
import time
import numpy as np

import debug

def train(env, agent, n_steps, log_dir = None, n_videos = 0):
    log_dir = __getenv("LOG_DIR") if log_dir is None else log_dir
    if n_videos == 0 and __getenv("N_VIDEOS") is not None:
        n_videos = max(0, int(__getenv("N_VIDEOS")))

    video = []
    if log_dir is not None:
        import gym.wrappers
        env = gym.wrappers.Monitor(
            env,
            log_dir,
            video_callable = lambda x: len(video) > 0 and video.pop(),
            force = True
        )

    history = []
    obs, reward, done = env.reset(), 0.0, False
    ep_r, ep_steps = 0.0, 0
    train_time, agent_time = time.time(), 0.0

    for t in range(n_steps):
        if (t+1) % max(10, ((n_steps+n_videos+1) // (n_videos+1))) == 0:
            video.append(True)

        step_time = time.time()
        action = agent.step(obs, reward, done)
        agent_time += time.time() - step_time

        if done:
            history.append([t + 1, ep_r, ep_steps])
            obs, reward, done = env.reset(), 0.0, False
            ep_r, ep_steps = 0.0, 0
        else:
            obs, reward, done, _ = env.step(action)
            ep_r += reward
            ep_steps += 1

    train_time = time.time() - train_time
    sys.stderr.write(
        "Finished in %.2fs: %.2f steps/s, %.1f%% agent time\n" %
        (train_time, n_steps/train_time, 100.0*agent_time/train_time)
    )
    sys.stderr.flush()

    if log_dir is not None:
        env.close()
        np.savetxt(
            log_dir + "/results.csv",
            history,
            fmt = "%10d %10.2f %10d",
            header = "    step  ep_reward   ep_steps"
        )

def get_run_args():
    import sys
    args = dict()
    for a in ("_".join(sys.argv[1:])).split("_"):
        if ":" not in a:
            continue
        a, val = a.split(":", 1)
        if val[0] in "0123456789-":
            if "." in val:
                args[a] = float(val)
            elif len(val) >= 1:
                args[a] = int(val)
        else:
            args[a] = val
    return args

def plot_results(*result_files):
    plot_file = __getenv("PLOT_FILE")

    import matplotlib
    if plot_file is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches
    fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
    fig.tight_layout()
    ax1.set_ylabel("reward/episode")
    ax2.set_ylabel("steps/episode")
    ax2.set_xlabel("step")
    ax1.grid()
    ax2.grid()
    patches = []

    for f, color in zip(result_files, "brgcmy"):
        data = np.genfromtxt(f, names=True)
        __add_plot(ax1, data["step"], data["ep_reward"], color)
        __add_plot(ax2, data["step"], data["ep_steps"], color)
        patches.append(matplotlib.patches.Patch(color=color))

    plt.figlegend(
        handles=patches,
        labels=map(__plot_name, result_files),
        loc=3
    )

    if plot_file is None:
        plt.show()
    else:
        fig.set_size_inches(8, 6)
        fig.savefig(plot_file, dpi=100)

def __plot_name(path):
    if path.endswith("/results.csv"):
        path = path[:-12]
    path = os.path.basename(path)
    return path.replace("_", " ")

def __batch_avg(data, batch):
    n = len(data) // batch
    data = data[0:batch*n]
    data = data.reshape((n, batch) + data.shape[1:])
    return np.mean(data, axis=1)

def __add_plot(ax, x, y, color):
    xy = np.vstack((x, y)).T
    ax.plot(xy[:,0], xy[:,1], color + ".", alpha=0.2, zorder=10)
    if len(xy) >= 20:
        xy = __batch_avg(xy, max(5, len(xy) // 50))
    ax.plot(xy[:,0], xy[:,1], "w-", linewidth=4, zorder=11)
    ax.plot(xy[:,0], xy[:,1], color + "-", linewidth=2, zorder=12)

def __getenv(name):
    if name in os.environ and len(os.environ[name]) >= 1:
        return os.environ[name]
    return None

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
