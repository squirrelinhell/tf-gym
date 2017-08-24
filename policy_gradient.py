#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tf_dist
import gym
import gym.spaces

import utils

def affine(x, out_dim):
    assert len(x.shape.as_list()) == 2
    w = tf.Variable(tf.truncated_normal(
        stddev = 0.1,
        shape = [int(x.shape[1]), out_dim],
        dtype = x.dtype
    ))
    b = tf.Variable(tf.truncated_normal(
        stddev = 0.1,
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
    def __init__(self, o_space, a_space, lr=0.02, eps=0.0001):
        obs = tf.placeholder(tf.float32, o_space)

        # Build graph
        layer = tf.reshape(obs, [1, -1])
        layer = affine(layer, 2 * np.prod(o_space))
        layer = tf.nn.relu(layer)

        if isinstance(a_space, int):
            # Discrete action space
            logdist = affine(layer, a_space)
            action = tf.to_int32(tf.multinomial(logdist, 1))[0][0]
            log_prob = tf.log(tf.nn.softmax(logdist[0])[action])
        else:
            # Continuous action space
            num_params = 2 * np.prod(a_space)
            params = affine(layer, num_params)
            params = tf.reshape(params, (2,) + a_space)
            gauss = tf_dist.Normal(params[0], params[1])
            action = tf.stop_gradient(gauss.sample())
            log_prob = gauss.log_prob(action)

        # Compute gradient
        params = tf.trainable_variables()
        elasticity = gradient(log_prob, params)
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

def run(env="CartPole-v1", steps=50000, end_reward=None,
        **kwargs):
    env = gym.make(env)
    env = utils.wrappers.Log(env)
    env = utils.wrappers.EndlessEpisode(env, end_reward)

    a_space = env.action_space
    if isinstance(a_space, gym.spaces.Discrete):
        a_space = a_space.n
    elif isinstance(a_space, gym.spaces.Box):
        env = utils.wrappers.UnboundedActions(env)
        a_space = a_space.shape
    else:
        raise ValueError("Unsupported action space")

    agent = PolicyGradientAgent(
        o_space=env.observation_space.shape,
        a_space=a_space,
        **kwargs
    )

    utils.train.thread(env, agent, steps)

if __name__ == "__main__":
    import sys
    run(**utils.train.parse_args(*sys.argv[1:]))
