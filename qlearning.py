#!/usr/bin/env python3

import numpy as np

import agent
import train
import debug

class QLearning(agent.Agent):
    def __init__(self, n_states, n_actions, lr = 0.03, discount = 0.9):
        self.v = np.random.rand(n_states, n_actions) * 0.1 - 0.05
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

    def __str__(self):
        return str(np.round(self.v, 2))

def run():
    import gym
    env = gym.make('FrozenLake-v0')
    agt = QLearning(env.observation_space.n, env.action_space.n)
    train.train(env, agt)

if __name__ == "__main__":
    run()
