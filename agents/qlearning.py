
import numpy as np

from agents import Agent

class QLearning(Agent):
    def __init__(self, o_space, a_space, lr = 0.03, discount = 0.9):
        self.v = np.random.rand(o_space.n, a_space.n) * 0.1 - 0.05
        self.last = None
        self.lr = lr
        self.discount = discount

    def step(self, obs, reward, done):
        if done:
            reward = 1.0 if reward > 0.0 else -1.0
        else:
            reward = 0.0

        if self.last is not None:
            self.v[self.last] *= 1.0 - self.lr
            self.v[self.last] += self.lr * (
                reward + self.discount * np.max(self.v[obs])
            )

        self.last = obs, self.policy(obs)
        return self.last[1]

    def policy(self, obs):
        if np.random.rand() < 0.001:
            return np.random.randint(self.v.shape[1])
        return np.argmax(self.v[obs])
