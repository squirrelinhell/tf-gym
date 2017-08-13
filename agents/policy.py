
import numpy as np
import tensorflow as tf

from agents import Agent

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
def gradient(x):
    vs = []
    gs = []
    pos = 0, 0
    for v in tf.trainable_variables():
        g = tf.gradients(x, v)[0]
        if g is None:
            gs.append(tf.zeros([tf.size(v)], v.dtype))
        else:
            gs.append(tf.reshape(g, [-1]))
        pos = pos[1], pos[1] + tf.size(v)
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

class PolicyAgent(Agent):
    def __init__(self, o_space, a_space,
            discount = 0.9, horizon = 500, batch = 128,
            lr = 0.02, eps = 0.0001,
            normalize_adv = 0.0, normalize_obs = 0.0,
            end_penalty = -100, hidden_layer = 8):
        self.discount = discount
        self.horizon = horizon
        self.batch = batch
        self.normalize_adv = running_normalize(lr = normalize_adv)
        self.normalize_obs = running_normalize(lr = normalize_obs)
        self.end_penalty = end_penalty

        # Policy network
        self.obs = tf.placeholder(tf.float32, o_space.shape)
        y = self.obs
        y = linear(y, hidden_layer)
        y = bias(y)
        y = tf.nn.relu(y)
        y = linear(y, a_space.n)
        y = bias(y)
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
        obs = self.normalize_obs(obs)
        if done:
            reward = self.end_penalty

        action, elasticity = self.sess.run(
            [self.action, self.elasticity],
            feed_dict = {self.obs: obs}
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

        advans = self.normalize_adv(
            [self._advantage(t) for t in range(self.batch)],
            avg=0
        )
        elasts = [h[1] for h in self.history[0:self.batch]]
        self.history = self.history[self.batch:]

        self.sess.run(
            self.grad_ascend,
            feed_dict = {self.grad_in: np.dot(advans, elasts)}
        )
