
import numpy as np
import tensorflow as tf

import agents

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

class PolicyAgent(agents.Agent):
    def __init__(self, o_space, a_space,
            discount = 0.9, horizon = 500, batch = 128,
            lr = 0.02, eps = 0.0001, normalize = "meanstd",
            endpenalty = -100, hiddenlayer = 8):
        self.discount = discount
        self.horizon = horizon
        self.batch = batch
        self.normalize = {
            "off": lambda x: x,
            "mean": lambda x: x - x.mean(),
            "meanstd": lambda x: (lambda y: y - y.std())(x - x.mean()),
        }[normalize]
        self.endpenalty = endpenalty

        # Policy network
        self.obs = tf.placeholder(tf.float32, o_space.shape)
        y = tf.reshape(self.obs, [1, -1])
        y = dense(y, hiddenlayer)
        y = tf.nn.relu(y)
        y = dense(y, a_space.n)
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

        self.history = []

    def step(self, obs, reward, done):
        if done:
            reward = self.endpenalty

        action, elasticity = self.sess.run(
            [self.action, self.elasticity],
            feed_dict={self.obs: obs}
        )

        self.history.append((reward, elasticity))
        self._learn()

        return action

    def _advantage(self, t):
        rs = self.history[t+1:t+1+self.horizon]
        assert len(rs) == self.horizon

        sum_r = 0.0
        for r, _ in reversed(rs):
            sum_r *= self.discount
            sum_r += r
        return sum_r

    def _learn(self):
        if len(self.history) < self.horizon + self.batch:
            return

        advans = [self._advantage(t) for t in range(self.batch)]
        advans = self.normalize(np.array(advans))
        elasts = [h[1] for h in self.history[0:self.batch]]
        self.history = self.history[self.batch:]

        self.sess.run(
            self.grad_ascend,
            feed_dict = {
                self.grad_in: np.dot(advans, elasts)
            }
        )