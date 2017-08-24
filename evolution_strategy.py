#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import gym
import gym.spaces

import utils

class NoisyParameters:
    def __init__(self, dtype=tf.float32):
        noise_stddev = tf.placeholder(dtype, [])

        params = []
        noise = []
        splits = [0]

        def normal(shape, stddev):
            p = tf.Variable(tf.truncated_normal(
                shape, stddev=stddev, dtype=dtype
            ))
            n = tf.Variable(tf.random_normal(
                shape, stddev=noise_stddev, dtype=dtype
            ))
            params.append(p)
            noise.append(n)
            splits.append(splits[-1] + tf.size(p))
            return p + n

        def prepare(sess, ctl):
            sess.run(tf.variables_initializer(params))

            init_noise = tf.variables_initializer(noise)
            noise_v = tf.concat(
                [tf.reshape(n, [-1]) for n in noise], axis=0
            )
            ctl.init_noise = lambda stddev: (
                sess.run(init_noise, feed_dict={noise_stddev: stddev}),
                sess.run(noise_v)
            )[1]

            inp = tf.placeholder(noise_v.dtype, noise_v.shape)
            add_to_params = [
                tf.assign_add(p, tf.reshape(inp[start:end], p.shape))
                for p, start, end in zip(params, splits, splits[1:])
            ]
            ctl.add_to_params = lambda feed_inp: sess.run(
                add_to_params,
                feed_dict={inp: feed_inp}
            )

        self.normal = normal
        self.prepare = prepare

class PolicyNetwork:
    def __init__(self, o_space, a_space, hidden_layer=100):
        obs = tf.placeholder(tf.float32, o_space)
        params = NoisyParameters()

        def linear(x, out_dim):
            assert len(x.shape.as_list()) == 2
            in_dim = int(x.shape[1])
            w = params.normal([in_dim, out_dim], 1.0 / np.sqrt(in_dim))
            return tf.matmul(x, w)

        v = tf.reshape(obs, [1, -1])
        v = linear(v, hidden_layer)
        v = tf.tanh(v)
        v = linear(v, np.prod(a_space))
        v = tf.tanh(v)
        action = tf.reshape(v, a_space)

        sess = tf.Session()
        params.prepare(sess, self)

        self.get_action = lambda feed_obs: sess.run(
            action,
            feed_dict = {obs: feed_obs}
        )

class EvolutionStrategyAgent(utils.train.Agent):
    def __init__(self, population=20, lr=0.07, noise=0.1,
            **kwargs):
        net = PolicyNetwork(**kwargs)
        rewards = [0.0]
        noise_vs = [net.init_noise(noise)]

        def normalized(a):
            a = np.asarray(a)
            a -= a.mean()
            return a / a.std()

        def learn():
            nonlocal rewards, noise_vs
            if len(noise_vs) < population:
                return

            grad = normalized(np.dot(rewards, noise_vs))
            net.add_to_params(lr * grad)

            rewards = []
            noise_vs = []

        def take_reward(reward, episode_end):
            rewards[-1] += reward
            if episode_end:
                learn()
                rewards.append(0.0)
                noise_vs.append(net.init_noise(noise))

        self.next_action = net.get_action
        self.take_reward = take_reward

def run(env="BipedalWalker-v2", steps=1000000, ep_limit=300, **kwargs):
    env = gym.make(env)
    env = utils.wrappers.LimitedEpisode(env, ep_limit)
    env = utils.wrappers.Log(env)
    env = utils.wrappers.UnboundedActions(env)

    agent = EvolutionStrategyAgent(
        o_space=env.observation_space.shape,
        a_space=env.action_space.shape,
        **kwargs
    )

    utils.train.thread(env, agent, steps)

if __name__ == "__main__":
    import sys
    run(**utils.train.parse_args(*sys.argv[1:]))
