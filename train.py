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
    obs, reward, done = env.reset(), 0.0, False
    ep_r, ep_steps = 0.0, 0
    train_time, agent_time = time.time(), 0.0

    for t in range(steps):
        if on_progress is not None:
            on_progress(t)

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
        (train_time, steps/train_time, 100.0*agent_time/train_time)
    )
    sys.stderr.flush()

    return history

def _auto_train(env, agent, steps = 50000,
        logdir = None, n_videos = None, **args):
    logdir = _getenv("LOG_DIR", logdir)
    n_videos = _getenv("N_VIDEOS", n_videos)
    n_videos = 0 if n_videos is None else int(n_videos)

    if not isinstance(env, gym.core.Env):
        env = gym.make(env)

    agent = _auto_build_agent(env, agent, **args)

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

def _auto_build_agent(env, agent, **args):
    if isinstance(agent, agents.Agent):
        assert len(args) == 0
        return agent

    args["o_space"] = env.observation_space
    args["a_space"] = env.action_space

    module = importlib.import_module("agents." + agent)
    for _, cls in module.__dict__.items():
        if _is_strict_subclass(cls, agents.Agent):
            return cls(**args)
    raise ValueError("No subclasses of Agent found in " + agent)

def _is_strict_subclass(a, b):
    if not isinstance(a, type):
        return False
    if not issubclass(a, b):
        return False
    return a != b

def _getenv(name, default = ""):
    if name in os.environ and len(os.environ[name]) >= 1:
        return os.environ[name]
    return default

def _parse_args(*args):
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
        _auto_train(**_parse_args(*sys.argv[1:]))
    else:
        sys.stderr.write("\nUsage:\n\n")
        sys.stderr.write("\t" + "train.py env:<name> agent:<name>\n")
        sys.stderr.write("\n")
        sys.exit(1)

if __name__ == "__main__":
    run()
