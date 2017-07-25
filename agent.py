
from utils import *

class Agent:
    def __str__(self):
        return "<Agent>"

    def step(self, obs, reward, done):
        raise NotImplementedError("Agent requires a step() method")

    def train(self, env, n_steps = 1000000):
        live_plot(self.__train_iter(env, n_steps), n_steps)

    def __train_iter(self, env, n_steps):
        obs, reward, done = env.reset(), 0.0, False
        episodes, r_sum  = 0, 0.0

        for t in range(1, n_steps + 1):
            r_sum += reward
            action = self.step(obs, reward, done)

            if done:
                episodes += 1
                obs, reward, done = env.reset(), 0.0, False
            else:
                obs, reward, done, _ = env.step(action)

            if episodes >= 100:
                yield t, r_sum / episodes
                episodes, r_sum = 0, 0.0
