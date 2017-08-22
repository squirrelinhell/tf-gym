#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from lib import debug, train, wrappers

def tf_scope(f):
    def f_with_scope(*args, **kwargs):
        with tf.name_scope(f.__name__):
            return f(*args, **kwargs)
    return f_with_scope

@tf_scope
def linear(x, out_dim):
    in_dim = np.prod([
        1 if v is None else v
        for v in x.shape.as_list()
    ])
    x = tf.reshape(x, [-1, in_dim])
    w = tf.Variable(tf.truncated_normal(
        stddev = 0.1,
        shape = [in_dim, out_dim],
        dtype = x.dtype
    ))
    return tf.matmul(x, w)

@tf_scope
def bias(x):
    b = tf.Variable(tf.truncated_normal(
        stddev = 0.1,
        shape = x.shape,
        dtype = x.dtype
    ))
    return x + b

@tf_scope
def gradient(var, params):
    ret = []
    for p in params:
        g = tf.gradients(var, p)[0]
        if g is None:
            ret.append(tf.zeros([tf.size(p)], p.dtype))
        else:
            ret.append(tf.reshape(g, [-1]))
    return tf.concat(ret, axis=0)

@tf_scope
def split_gradient(grad, params):
    ret = []
    start, end = 0, 0
    for p in params:
        start, end = end, end + tf.size(p)
        ret.append((tf.reshape(grad[start:end], p.shape), p))
    return ret

class Network:
    def __init__(self, o_space, n_actions,
            hidden_layer=8, lr=0.02, eps=0.0001):
        obs = tf.placeholder(tf.float32, o_space)

        # Policy network
        layer = tf.nn.relu(bias(linear(obs, hidden_layer)))
        policy = bias(linear(layer, n_actions))
        action = tf.to_int32(tf.multinomial(policy, 1))[0][0]
        policy = tf.nn.softmax(policy[0])

        # Compute gradient
        params = tf.trainable_variables()
        elasticity = gradient(tf.log(policy[action]), params)

        # Apply gradient
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

class PolicyAgent(train.Agent):
    def __init__(self,
            discount=0.9, horizon=500,
            batch=128, end_penalty=-100,
            normalize_adv=0.0, normalize_obs=0.0,
            **kwargs):
        normalize_adv = running_normalize(lr=normalize_adv)
        normalize_obs = running_normalize(lr=normalize_obs)

        net = Network(**kwargs)
        rewards = []
        elasts = []

        def advantage(time):
            sum_r = 0.0
            for t1 in reversed(range(time, time + horizon)):
                sum_r *= discount
                sum_r += rewards[t1]
            return sum_r

        def learn():
            nonlocal rewards, elasts
            if len(rewards) < batch + horizon:
                return

            advantages = [advantage(t) for t in range(batch)]
            net.grad_ascend(np.dot(advantages, elasts[0:batch]))

            rewards = rewards[batch:]
            elasts = elasts[batch:]

        def next_action(obs):
            obs = normalize_obs(obs)
            action, elasticity = net.process_step(obs)

            elasts.append(elasticity)
            return action

        def take_reward(reward, episode_end):
            if episode_end:
                reward = end_penalty

            rewards.append(reward)
            assert len(rewards) == len(elasts)

            learn()

        self.next_action = next_action
        self.take_reward = take_reward

def run(env = "CartPole-v1", *args, **kwargs):
    import gym
    env = wrappers.Log(gym.make(env))
    agent = PolicyAgent(
        o_space=env.observation_space.shape,
        n_actions=env.action_space.n,
        *args, **kwargs
    )
    train.thread(env, agent, 50000)

if __name__ == "__main__":
    import sys
    run(**train.parse_args(*sys.argv[1:]))
