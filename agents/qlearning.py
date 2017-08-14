
import numpy as np

from agents import Agent

class QLearning(Agent):
    def __init__(self, o_space, a_space, lr = 0.03, discount = 0.9):
        self.v = np.random.rand(o_space.n, a_space.n) * 0.1 - 0.05
        self.last = None
        self.lr = lr
        self.discount = discount

    def next_action(self, obs):
        if self.last is not None:
            self.v[self.last] *= 1.0 - self.lr
            self.v[self.last] += self.lr * (
                self.reward + self.discount * np.max(self.v[obs])
            )

        if np.random.rand() < 0.001:
            action = np.random.randint(self.v.shape[1])
        else:
            action = np.argmax(self.v[obs])

        self.last = obs, action
        return action

    def take_reward(self, reward, episode_end):
        if episode_end:
            self.reward = 1.0 if reward > 0.0 else -1.0
        else:
            self.reward = 0.0
