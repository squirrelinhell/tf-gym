
import tensorflow as tf

def tf_scope(f):
    def f_with_scope(*args, **kwargs):
        with tf.name_scope(f.__name__):
            return f(*args, **kwargs)
    return f_with_scope

@tf_scope
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
