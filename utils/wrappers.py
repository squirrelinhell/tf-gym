
import os
import sys
import gym
import gym.wrappers
import gym.spaces
import numpy as np

class EndlessEpisode(gym.Wrapper):
    def __init__(self, env, end_reward=None):
        super().__init__(env)

        def do_step(action):
            obs, reward, done, info = self.env._step(action)
            if done:
                done = False
                obs = self.env._reset()
                if end_reward is not None:
                    reward = end_reward
            return obs, reward, done, info

        self._step = do_step

class UnboundedActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        low = self.action_space.low
        diff = self.action_space.high - low
        assert (diff > 0.001).all()
        assert (diff < 1000).all()

        def do_step(action):
            action = np.asarray(action)
            action = np.abs((action - 1.0) % 4.0 - 2.0) * 0.5
            action = diff * action + low
            return self.env._step(action)

        self._step = do_step
        self.action_space = gym.spaces.Box(-np.inf, np.inf, low.shape)

class Log(gym.Wrapper):
    def __init__(self, env, log_dir="", video_every=0):
        if log_dir == "" and "LOG_DIR" in os.environ:
            log_dir = os.environ["LOG_DIR"]
        if video_every < 1 and "VIDEO_EVERY" in os.environ:
            video_every = int(os.environ["VIDEO_EVERY"])

        t = 0
        ep_r, ep_steps = 0.0, 0
        video_wanted = False
        history = []

        def should_record_video(_):
            nonlocal video_wanted
            r = video_wanted
            video_wanted = False
            return r

        if len(log_dir) >= 1 and video_every >= 1:
            env = gym.wrappers.Monitor(
                env,
                log_dir,
                video_callable=should_record_video,
                force = True
            )

        super().__init__(env)

        def do_step(action):
            nonlocal t, ep_r, ep_steps, video_wanted
            obs, reward, done, info = self.env.step(action)

            if video_every >= 1 and t > 0 and t % video_every == 0:
                video_wanted = True

            t += 1
            ep_r += reward
            ep_steps += 1

            if done:
                history.append([t, ep_r, ep_steps])
                ep_r, ep_steps = 0.0, 0

            return obs, reward, done, info

        def do_close():
            self.env._close()
            if len(log_dir) >= 1 and len(history) >= 1:
                os.makedirs(log_dir, exist_ok=True)
                np.savetxt(
                    log_dir + "/episodes.csv",
                    history,
                    fmt = "%10d %10.2f %10d",
                    header = "    step  ep_reward   ep_steps"
                )

        self._step = do_step
        self._close = do_close

class Trace(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        def print_calls(f):
            def wrapped_f(*args, **kwargs):
                ret = f(*args, **kwargs)
                sys.stderr.write(f.__name__
                    + "(" + ", ".join(map(str, args))
                    + ") -> " + str(ret) + "\n")
                sys.stderr.flush()
                return ret
            return wrapped_f

        self._step = print_calls(self.env._step)
        self._reset = print_calls(self.env._reset)
