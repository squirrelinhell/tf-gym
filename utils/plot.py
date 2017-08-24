#!/usr/bin/env python3

import os
import sys
import time
import numpy as np

def plot_csv(*csv_files):
    plot_file = _getenv("PLOT_FILE")

    import matplotlib
    if plot_file is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches
    patches = []

    cols = list(np.genfromtxt(csv_files[0], names=True).dtype.names)
    assert len(cols) >= 2

    fig, axes = plt.subplots(nrows=len(cols)-1, sharex=True)
    axes[-1].set_xlabel(cols[0])
    for col, ax in zip(cols[1:], axes):
        ax.set_ylabel(col)
        ax.grid()

    for file_name, color in zip(csv_files, "brgcmy"):
        patches.append(matplotlib.patches.Patch(color=color))
        data = np.genfromtxt(file_name, names=True)

        for col, ax in zip(cols[1:], axes):
            _add_plot(ax, data[cols[0]], data[col], color)

    labels = list(map(os.path.basename, csv_files))
    if len(set(labels)) < len(csv_files):
        labels = ["/".join(n.split("/")[-2:]) for n in csv_files]
    plt.figlegend(handles=patches, labels=labels, loc=3)

    if plot_file is None:
        plt.show()
    else:
        fig.set_size_inches(10, 8)
        fig.savefig(plot_file, dpi=100)

def _running_mean(data, window):
    sums = np.cumsum(data, 0)
    return (sums[window:] - sums[:-window]) / window

def _add_plot(ax, x, y, color):
    xy = np.vstack((x, y)).T
    ax.plot(xy[:,0], xy[:,1], color + ".", alpha=0.2, zorder=10)
    if len(xy) >= 20:
        xy = _running_mean(xy, max(5, len(xy) // 10))
    ax.plot(xy[:,0], xy[:,1], "w-", linewidth=4, zorder=11)
    ax.plot(xy[:,0], xy[:,1], color + "-", linewidth=2, zorder=12)

def _getenv(name):
    if name in os.environ and len(os.environ[name]) >= 1:
        return os.environ[name]
    return None

def run():
    if len(sys.argv) >= 2:
        plot_csv(*sys.argv[1:])
    else:
        sys.stderr.write("\nUsage:\n\n")
        sys.stderr.write("\t" + "plot.py <f1.csv> [ ... ]\n")
        sys.stderr.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    run()
