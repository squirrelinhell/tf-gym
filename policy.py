#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

import agent
import train
import debug

def dense(x, out_dim, name = "dense"):
    with tf.name_scope(name):
        in_dim = np.prod([
            1 if v is None else v
            for v in x.shape.as_list()
        ])
        x = tf.reshape(x, [-1, in_dim])
        b = tf.Variable(
            tf.constant(0.1, shape=[out_dim], dtype=tf.float32),
            name = "b"
        )
        w = tf.Variable(
            tf.truncated_normal([in_dim, out_dim], stddev=0.1,
                dtype=tf.float32),
            name = "w"
        )
        return tf.matmul(x, w) + b

class PolicyAgent(agent.Agent):
    def __init__(self, obs_shape, n_actions,
            discount = 0.9, histlen = 100, batch = 128,
            lr = 0.01, eps = 0.0001,
            endpenalty = -100, hiddenlayer = 8):
        self.n_actions = n_actions
        self.discount = discount
        self.histlen = histlen
        self.batch = batch
        self.endpenalty = endpenalty

        # Policy network
        self.obs = tf.placeholder(tf.float32, obs_shape)
        self.policy = self.__policy(self.obs, hiddenlayer)

        # Compute gradients
        self.reward = tf.placeholder(tf.float32, [])
        self.action = tf.placeholder(tf.int32, [])
        loss = -tf.log(self.policy[self.action]) * self.reward
        opt = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        )
        grads = opt.compute_gradients(loss)
        self.grads = [t[0] for t in grads]

        # Apply gradients
        grads_in = [
            (tf.placeholder(tf.float32, g.shape), v) for g, v in grads
        ]
        self.grads_in = [t[0] for t in grads_in]
        self.grads_apply = opt.apply_gradients(grads_in)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.hist_buffer = []
        self.grad_buffer = []

    def __policy(self, x, hiddenlayer):
        x = dense(x, hiddenlayer)
        x = tf.nn.relu(x)
        x = dense(x, self.n_actions)
        x = tf.reshape(x, [-1])
        return tf.nn.softmax(x)

    def step(self, obs, reward, done):
        if done:
            reward = self.endpenalty

        policy = self.sess.run(
            self.policy,
            feed_dict={self.obs: obs}
        )
        action = np.random.choice(self.n_actions, p=policy)
        self.hist_buffer.append((obs, reward, action))

        self.__compute_gradients()
        self.__apply_gradients()

        return action

    def __compute_gradients(self):
        if len(self.hist_buffer) < self.histlen:
            return

        obs, reward, action = self.hist_buffer[0]
        self.hist_buffer = self.hist_buffer[1:]

        sum_r = 0.0
        for _, r, _ in reversed(self.hist_buffer):
            sum_r *= self.discount
            sum_r += r

        grads = self.sess.run(
            self.grads,
            feed_dict={
                self.obs: obs,
                self.reward: sum_r,
                self.action: action
            }
        )
        self.grad_buffer.append(grads)

    def __apply_gradients(self):
        if len(self.grad_buffer) < self.batch:
            return

        grads = self.grad_buffer[0]
        for add in self.grad_buffer[1:]:
            for g, a in zip(grads, add):
                g += a

        self.grad_buffer = []
        grads = self.sess.run(
            self.grads_apply,
            feed_dict=dict(zip(self.grads_in, grads))
        )

    def __str__(self):
        return str(np.round(self.v, 2))

def run(env = "CartPole-v1", **args):
    import gym
    env = gym.make(env)

    agt = PolicyAgent(
        env.observation_space.shape,
        env.action_space.n,
        **args
    )

    train.train(env, agt, 50000)

if __name__ == "__main__":
    run(**train.get_run_args())
