#!/usr/bin/env python3

import os
import sys
import time
import importlib
import gym
import gym.wrappers
import numpy as np

import agents
import debug

def train(env, agent, steps, on_progress = None):
    history = []
    obs, ep_r, ep_steps = env.reset(), 0.0, 0
    timer = Timer()

    for t in range(steps):
        if on_progress is not None:
            on_progress(t)

        action = timer.call(agent.next_action, obs)

        obs, reward, done, _ = env.step(action)
        ep_r += reward
        ep_steps += 1

        timer.call(agent.take_reward, reward, episode_end = done)

        if done:
            history.append([t + 1, ep_r, ep_steps])
            obs, ep_r, ep_steps = env.reset(), 0.0, 0

    timer.end()
    sys.stderr.write(
        "Finished in %.2fs: %.2f steps/s, %.1f%% agent time\n" %
        (timer.total, steps/timer.total, 100.0*timer.calls/timer.total)
    )
    sys.stderr.flush()

    return history

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

def auto_train(env, agent, steps = 50000,
        logdir = None, n_videos = None, **args):
    logdir = get_env("LOG_DIR", logdir)
    n_videos = get_env("N_VIDEOS", n_videos)
    n_videos = 0 if n_videos is None else int(n_videos)

    if not isinstance(env, gym.core.Env):
        env = gym.make(env)

    agent = auto_build_agent(env, agent, **args)

    videos = []
    on_progress = lambda t: \
        videos.append(True) \
        if (t+1) % max(10, ((steps+n_videos+1) // (n_videos+1))) == 0 \
        else None

    if logdir is not None:
        env = gym.wrappers.Monitor(
            env,
            logdir,
            video_callable = lambda x: len(videos) > 0 and videos.pop(),
            force = True
        )

    history = train(env, agent, steps, on_progress)

    if logdir is not None:
        env.close()
        np.savetxt(
            logdir + "/results.csv",
            history,
            fmt = "%10d %10.2f %10d",
            header = "    step  ep_reward   ep_steps"
        )

def auto_build_agent(env, agent, **args):
    if isinstance(agent, agents.Agent):
        assert len(args) == 0
        return agent

    args["o_space"] = env.observation_space
    args["a_space"] = env.action_space

    module = importlib.import_module("agents." + agent)
    for _, cls in module.__dict__.items():
        if is_strict_subclass(cls, agents.Agent):
            return cls(**args)
    raise ValueError("No subclasses of Agent found in " + agent)

def is_strict_subclass(a, b):
    if not isinstance(a, type):
        return False
    if not issubclass(a, b):
        return False
    return a != b

def get_env(name, default = ""):
    if name in os.environ and len(os.environ[name]) >= 1:
        return os.environ[name]
    return default

def parse_args(*args):
    result = dict()
    for a in ("_".join(args)).split("_"):
        if ":" not in a:
            continue
        a, val = a.split(":", 1)
        a = a.replace("-","_")
        if val[0] in "0123456789-":
            if "." in val:
                result[a] = float(val)
            elif len(val) >= 1:
                result[a] = int(val)
        else:
            result[a] = val
    return result

def run():
    if len(sys.argv) >= 2:
        auto_train(**parse_args(*sys.argv[1:]))
    else:
        sys.stderr.write("\nUsage:\n\n")
        sys.stderr.write("\t" + "train.py env:<name> agent:<name>\n")
        sys.stderr.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    run()
