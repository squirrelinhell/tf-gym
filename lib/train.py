
import sys
import time

class Agent:
    def next_action(self, obs):
        raise NotImplementedError("Not implemented: next_action()")

    def take_reward(self, reward, episode_end):
        pass

def thread(env, agent, steps=50000, print_stats=True):
    class Timer:
        def __init__(self):
            self.start = time.time()
            self.calls = 0.0
        def call(self, f, *args, **kwargs):
            t0 = time.time()
            ret = f(*args, **kwargs)
            self.calls += time.time() - t0
            return ret
        def end(self):
            self.total = time.time() - self.start

    history = []
    obs = env.reset()
    timer = Timer()

    for t in range(steps):
        action = timer.call(agent.next_action, obs)

        obs, reward, done, _ = env.step(action)

        timer.call(agent.take_reward, reward, episode_end = done)

        if done:
            obs = env.reset()

    timer.end()

    if print_stats:
        sys.stderr.write(
            "%d steps in %.2fs: %.2f steps/s, %.1f%% agent time\n" %
            (steps, timer.total, steps/timer.total,
                100.0 * timer.calls / timer.total)
        )
        sys.stderr.flush()

    return history

def parse_args(*args):
    result = dict()
    for arg in (",".join(args)).split(","):
        if not ":" in arg:
            continue
        name, value = arg.split(":", 1)
        name, value = name.strip(), value.strip()
        if value[0] in "0123456789-":
            if "." in value:
                result[name] = float(value)
            elif len(value) >= 1:
                result[name] = int(value)
        else:
            result[name] = value
    return result
