import tensorflow as tf
import tensorflow.contrib.slim as slim

# Xu, Bing, et al.
#"Empirical evaluation of rectified activations in convolutional network." arXiv preprint arXiv:1505.00853 (2015).
def lrelu(x, leak = 0.2, alt_relu_impl = False):
    with tf.variable_scope('lrelu'):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)

# Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky.
#"Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
def instance_norm(x, epsilon = 1e-5):
    with tf.variable_scope('instance_norm'):
        mean, var = tf.nn.moments(x, [1, 2], keep_dims = True)
        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out
