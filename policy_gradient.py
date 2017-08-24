#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tf_dist
import gym
import gym.spaces

import utils

def affine(x, out_dim):
    assert len(x.shape.as_list()) == 2
    in_dim = int(x.shape[1])
    stddev = 1 / np.sqrt(in_dim + 1)
    w = tf.Variable(tf.truncated_normal(
        stddev = stddev,
        shape = [in_dim, out_dim],
        dtype = x.dtype
    ))
    b = tf.Variable(tf.truncated_normal(
        stddev = stddev,
        shape = [out_dim],
        dtype = x.dtype
    ))
    return tf.matmul(x, w) + b

def gradient(var, params):
    ret = []
    for p in params:
        g = tf.gradients(var, p)[0]
        if g is None:
            ret.append(tf.zeros([tf.size(p)], p.dtype))
        else:
            ret.append(tf.reshape(g, [-1]))
    return tf.concat(ret, axis=0)

def split_gradient(flat, params):
    ret = []
    start, end = 0, 0
    for p in params:
        start, end = end, end + tf.size(p)
        ret.append((tf.reshape(flat[start:end], p.shape), p))
    return ret

class PolicyNetwork:
    def __init__(self, o_space, n_actions, lr=0.02, eps=0.0001):
        obs = tf.placeholder(tf.float32, o_space)

        # Policy
        layer = tf.reshape(obs, [1, -1])
        layer = affine(layer, 2 * np.prod(o_space))
        layer = tf.nn.relu(layer)
        policy = affine(layer, n_actions)
        action = tf.to_int32(tf.multinomial(policy, 1))[0][0]
        policy = tf.nn.softmax(policy[0])

        # Gradient
        params = tf.trainable_variables()
        elasticity = gradient(tf.log(policy[action]), params)
        grad_in = tf.placeholder(elasticity.dtype, elasticity.shape)
        grad_ascend = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        ).apply_gradients(split_gradient(-grad_in, params))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.process_step = lambda feed_obs: sess.run(
            [action, elasticity],
            feed_dict = {obs: feed_obs}
        )

        self.grad_ascend = lambda feed_grad: sess.run(
            grad_ascend,
            feed_dict = {grad_in: feed_grad}
        )

def running_normalize(lr = 0.0001):
    if lr < 0.00000001:
        return lambda x, **kwargs: x
    params = [0, 1]
    def update(i, value, avg):
        if avg is not None:
            value = np.mean(value, axis=avg)
        params[i] = params[i] * (1.0 - lr) + value * lr
        return params[i]
    def process(value, avg = None):
        value = np.asarray(value)
        value -= update(0, value, avg)
        stddev = np.sqrt(update(1, np.square(value), avg))
        return value / np.maximum(0.0001, stddev)
    return process

class PolicyGradientAgent(utils.train.Agent):
    def __init__(self,
            horizon=50, batch=128,
            normalize_adv=0.0, normalize_obs=0.0,
            **kwargs):
        normalize_adv = running_normalize(lr=normalize_adv)
        normalize_obs = running_normalize(lr=normalize_obs)

        net = PolicyNetwork(**kwargs)
        rewards = []
        elasts = []

        def advantage(time):
            sum_r = 0.0
            for t1 in reversed(range(time, time + horizon)):
                sum_r *= 1.0 - (1.0/horizon)
                sum_r += rewards[t1]
            return sum_r

        def learn():
            nonlocal rewards, elasts
            if len(rewards) < batch + horizon:
                return

            advantages = [advantage(t) for t in range(batch)]
            advantages = normalize_adv(advantages, avg=0)
            net.grad_ascend(np.dot(advantages, elasts[0:batch]))

            rewards = rewards[batch:]
            elasts = elasts[batch:]

        def next_action(obs):
            obs = normalize_obs(obs)
            action, elasticity = net.process_step(obs)

            elasts.append(elasticity)
            return action

        def take_reward(reward, episode_end):
            rewards.append(reward)
            assert len(rewards) == len(elasts)

            learn()

        self.next_action = next_action
        self.take_reward = take_reward

def run(env="CartPole-v1", steps=50000, end_reward=-100, **kwargs):
    env = gym.make(env)
    env = utils.wrappers.Log(env)
    env = utils.wrappers.EndlessEpisode(env, end_reward)

    agent = PolicyGradientAgent(
        o_space=env.observation_space.shape,
        n_actions=env.action_space.n,
        **kwargs
    )

    utils.train.thread(env, agent, steps)

if __name__ == "__main__":
    import sys
    run(**utils.train.parse_args(*sys.argv[1:]))
