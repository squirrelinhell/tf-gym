
import os
import sys

def monitored(env):
    import gym.wrappers
    n = 1
    while os.path.exists("/tmp/log-" + str(n)):
        n += 1
    return gym.wrappers.Monitor(
        env,
        "/tmp/log-" + str(n),
        video_callable = False
    )

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

def train(env, agent, n_steps = 1000000):
    live_plot(__train_iter(env, agent, n_steps), n_steps)

def __train_iter(env, agent, n_steps):
    obs, reward, done = env.reset(), 0.0, False
    episodes, r_sum = 0, 0.0

    for t in range(1, n_steps + 1):
        r_sum += reward
        action = agent.step(obs, reward, done)

        if done:
            episodes += 1
            obs, reward, done = env.reset(), 0.0, False
        else:
            obs, reward, done, _ = env.step(action)

        if episodes >= 100:
            yield t, r_sum / episodes
            episodes, r_sum = 0, 0.0
