
import os
import sys

def monitored(env):
    import gym.wrappers
    n = 1
    while os.path.exists("/tmp/log-" + str(n)):
        n += 1
    return gym.wrappers.Monitor(env, "/tmp/log-" + str(n))

def live_plot(xy, x_max = 1.0, y_max = 1.0):
    axes = None
    xs, ys = [], []
    for x, y in xy:
        sys.stdout.write("(%f, %f)\n" % (x, y))
        sys.stdout.flush()
        xs.append(x)
        ys.append(y)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
        if "DISPLAY" in os.environ and len(os.environ["DISPLAY"]) > 0:
            import matplotlib.pyplot as plt
            plt.ion()
            if axes is None:
                axes = plt.gca()
                axes.grid()
            axes.set_xlim([0, x_max])
            axes.set_ylim([0, y_max])
            plt.plot(xs, ys, marker='o', color='black')
            plt.pause(0.01)
