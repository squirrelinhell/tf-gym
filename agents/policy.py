
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
    def __init__(self,
            o_space, a_space,
            hidden_layer = 8,
            lr = 0.02, eps = 0.0001,
            value_grad = 0.0):
        obs = tf.placeholder(tf.float32, o_space.shape)

        # Policy network
        layer = tf.nn.relu(bias(linear(obs, hidden_layer)))
        policy = bias(linear(layer, a_space.n))
        action = tf.to_int32(tf.multinomial(policy, 1))[0][0]
        policy = tf.nn.softmax(policy[0])
        pred_value = bias(linear(layer, 1))[0][0]

        # Compute gradient
        params = tf.trainable_variables()
        grad = gradient(
            tf.log(policy[action]) + pred_value * value_grad,
            params
        )

        # Apply gradient
        grad_in = tf.placeholder(grad.dtype, grad.shape)
        grad_ascend = tf.train.AdamOptimizer(
            learning_rate = lr,
            epsilon = eps
        ).apply_gradients(split_gradient(-grad_in, params))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        def process_step(feed_obs):
            a, v, g = sess.run(
                [action, pred_value, grad],
                feed_dict = {obs: feed_obs}
            )
            return a, v if value_grad > 0.000001 else 0.0, g

        self.process_step = process_step
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

class PolicyAgent(Agent):
    def __init__(self,
            discount = 0.9, horizon = 500,
            batch = 128, end_penalty = -100,
            normalize_adv = 0.0, normalize_obs = 0.0,
            **kwargs):
        normalize_adv = running_normalize(lr = normalize_adv)
        normalize_obs = running_normalize(lr = normalize_obs)

        net = Network(**kwargs)
        history = []

        def advantage(time):
            sum_r = 0.0
            for t1 in reversed(range(time, time + horizon)):
                sum_r *= discount
                sum_r += history[t1]["reward"]
            return sum_r - history[time]["pred_value"]

        def learn():
            nonlocal history
            if len(history) < horizon + batch:
                return

            advantages = [advantage(t) for t in range(batch)]
            grads = [h["grad"] for h in history[0:batch]]
            history = history[batch:]

            net.grad_ascend(np.dot(advantages, grads))

        def next_action(obs):
            obs = normalize_obs(obs)
            action, pred_value, grad = net.process_step(obs)

            history.append({
                "pred_value": pred_value,
                "grad": grad,
            })
            return action

        def take_reward(reward, episode_end):
            if episode_end:
                reward = end_penalty

            history[-1]["reward"] = reward
            learn()

        self.next_action = next_action
        self.take_reward = take_reward
