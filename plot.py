#!/usr/bin/env python3

import os
import sys
import time
import numpy as np

import debug

def plot_results(*result_files):
    plot_file = _getenv("PLOT_FILE")

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
        _add_plot(ax1, data["step"], data["ep_reward"], color)
        _add_plot(ax2, data["step"], data["ep_steps"], color)
        patches.append(matplotlib.patches.Patch(color=color))

    plt.figlegend(
        handles=patches,
        labels=map(_plot_name, result_files),
        loc=3
    )

    if plot_file is None:
        plt.show()
    else:
        fig.set_size_inches(8, 6)
        fig.savefig(plot_file, dpi=100)

def _plot_name(path):
    if path.endswith("/results.csv"):
        path = path[:-12]
    path = os.path.basename(path)
    return path.replace("_", " ")

def _batch_avg(data, batch):
    n = len(data) // batch
    data = data[0:batch*n]
    data = data.reshape((n, batch) + data.shape[1:])
    return np.mean(data, axis=1)

def _add_plot(ax, x, y, color):
    xy = np.vstack((x, y)).T
    ax.plot(xy[:,0], xy[:,1], color + ".", alpha=0.2, zorder=10)
    if len(xy) >= 20:
        xy = _batch_avg(xy, max(5, len(xy) // 50))
    ax.plot(xy[:,0], xy[:,1], "w-", linewidth=4, zorder=11)
    ax.plot(xy[:,0], xy[:,1], color + "-", linewidth=2, zorder=12)

def _getenv(name):
    if name in os.environ and len(os.environ[name]) >= 1:
        return os.environ[name]
    return None

def run():
    if len(sys.argv) >= 2:
        plot_results(*sys.argv[1:])
    else:
        sys.stderr.write("\nUsage:\n\n")
        sys.stderr.write("\t" + "plot.py <results.csv> [ ... ]\n")
        sys.stderr.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    run()
