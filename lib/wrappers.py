
import os
import gym
import gym.wrappers
import numpy as np

class Log(gym.Wrapper):
    def __init__(self, env, log_dir="", video_every=0):
        if log_dir == "" and "LOG_DIR" in os.environ:
            log_dir = os.environ["LOG_DIR"]

        t = 0
        ep_r, ep_steps = 0.0, 0
        video_wanted = False
        history = []

        def should_record_video(_):
            nonlocal video_wanted
            r = video_wanted
            video_wanted = False
            return r

        if len(log_dir) >= 1:
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
                history.append([t + 1, ep_r, ep_steps])
                ep_r, ep_steps = 0.0, 0

            return obs, reward, done, info

        def do_close():
            self.env._close()
            if len(log_dir) >= 1:
                np.savetxt(
                    log_dir + "/results.csv",
                    history,
                    fmt = "%10d %10.2f %10d",
                    header = "    step  ep_reward   ep_steps"
                )

        self._step = do_step
        self._close = do_close
