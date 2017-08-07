#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import agent
import train
import debug

def dense(x, out_dim, name = "dense"):
    with tf.name_scope(name):
        w = tf.Variable(
            tf.truncated_normal([int(x.shape[1]), out_dim], stddev=0.1,
                dtype=tf.float32),
            name = "w"
        )
        b = tf.Variable(
            tf.constant(0.1, shape=[out_dim], dtype=tf.float32),
            name = "b"
        )
        return tf.matmul(x, w) + b

def gradient(x):
    vs = []
    gs = []
    pos = 0, 0
    for v in tf.trainable_variables():
        g = tf.gradients(x, v)[0]
        gs.append(tf.reshape(g, [-1]))
        pos = pos[1], pos[1] + tf.size(g)
        vs.append((pos, v))

    # Return a single concatenated vector and a function that
    # can split such vectors into a (grad, variable) list
    return (
        tf.concat(gs, axis=0),
        lambda g: [
            (tf.reshape(g[p1:p2], v.shape), v)
            for (p1, p2), v in vs
        ]
    )

class PolicyAgent(agent.Agent):
    def __init__(self, obs_shape, n_actions,
            discount = 0.9, histlen = 500, batch = 128,
            lr = 0.02, eps = 0.0001,
            endpenalty = -100, hiddenlayer = 8):
        self.discount = discount
        self.histlen = histlen
        self.batch = batch
        self.endpenalty = endpenalty

        # Policy network
        self.obs = tf.placeholder(tf.float32, obs_shape)
        y = tf.reshape(self.obs, [1, -1])
        y = dense(y, hiddenlayer)
        y = tf.nn.relu(y)
        y = dense(y, n_actions)
        self.action = tf.to_int32(tf.multinomial(y, 1))[0][0]
        y = tf.nn.softmax(y[0])
        self.elasticity, split_gradient = gradient(
            tf.log(y[self.action])
        )

        # Apply gradients
        self.grad_in = tf.placeholder(tf.float32, self.elasticity.shape)
        self.grad_ascend = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        ).apply_gradients(split_gradient(-self.grad_in))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.hist_buffer = []
        self.grad_buffer = []

    def step(self, obs, reward, done):
        if done:
            reward = self.endpenalty

        action, elasticity = self.sess.run(
            [self.action, self.elasticity],
            feed_dict={self.obs: obs}
        )
        self.hist_buffer.append((reward, elasticity))

        self.__compute_gradients()
        self.__apply_gradients()

        return action

    def __compute_gradients(self):
        if len(self.hist_buffer) < self.histlen:
            return

        reward, elasticity = self.hist_buffer[0]
        self.hist_buffer = self.hist_buffer[1:]

        sum_r = 0.0
        for r, _ in reversed(self.hist_buffer):
            sum_r *= self.discount
            sum_r += r

        self.grad_buffer.append(elasticity * sum_r)

    def __apply_gradients(self):
        if len(self.grad_buffer) < self.batch:
            return

        self.sess.run(
            self.grad_ascend,
            feed_dict = {
                self.grad_in: np.mean(self.grad_buffer, axis=0)
            }
        )
        self.grad_buffer = []

    def __str__(self):
        return str(np.round(self.v, 2))

def run(env = "CartPole-v1", trainsteps = 50000, **args):
    import gym
    env = gym.make(env)

    agt = PolicyAgent(
        env.observation_space.shape,
        env.action_space.n,
        **args
    )

    train.train(env, agt, trainsteps)

if __name__ == "__main__":
    run(**train.get_run_args())
