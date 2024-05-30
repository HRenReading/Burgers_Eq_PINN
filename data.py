import tensorflow as tf


def train(n, minv, maxv):
    data = tf.random.uniform((n, 2), minv, maxv)
    return data


def test(n, minv, maxv):
    data = tf.linspace(minv, maxv, n, axis=0)
    return data

def data_init(n, minv, maxv):
    xt0 = tf.random.uniform((n,1), minv, maxv)
    return tf.concat([xt0, tf.zeros_like(xt0)], axis=1)


def data_bc1(n, minv, maxv):
    tx0 = tf.random.uniform((n,1), minv, maxv)
    return tf.concat([tf.ones_like(tx0), tx0], axis=1)

def data_bc2(n, minv, maxv):
    tx0 = tf.random.uniform((n,1), minv, maxv)
    return tf.concat([-tf.ones_like(tx0), tx0], axis=1)

