import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

def weight_variable(shape):
    weights = tf.get_variable("weights", shape, initializer=tf.random_normal_initializer(mean = 0.0, stddev=0.02))
    #print weights.name
    return weights

def bias_variable(shape):
    biases = tf.get_variable("biases", shape, initializer=tf.constant_initializer(0.0))
    return biases

def conv2d(x, W, stride = 2, padding = "SAME"):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

def deconv2d(x, W, output_shape, stride = 2, padding="SAME"):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding=padding)

def max_pool(x, stride, padding="VALID"):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=padding)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def residual(input, name, input_channel, reuse):
    with tf.variable_scope(name, reuse) as scope:
        with tf.variable_scope('res_conv1', reuse) as scope:
            W_conv = weight_variable([ 3, 3, input_channel, input_channel ])
            conv = conv2d(input, W_conv, stride = 1)
            conv = inst_norm(conv)
            conv = tf.nn.relu(conv)

        with tf.variable_scope('res_conv2', reuse) as scope:
            W_conv = weight_variable([ 3, 3, input_channel, input_channel ])
            conv = conv2d(conv, W_conv, stride = 1)
            conv = inst_norm(conv)

        conv = conv + input

    return conv

def batch_norm(x, scope, epsilon=1e-5, momentum = 0.9, is_training = True):
    return tf.contrib.layers.batch_norm(x,
                    decay=momentum, 
                    updates_collections=None,
                    epsilon=epsilon,
                    scale=True,
                    is_training=is_training,
                    scope=scope)



def inst_norm(inputs, epsilon=1e-3, suffix=''):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
    http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.
    :param input_:
    input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.get_variable('INscale'+suffix,
            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift'+suffix,
            initializer=tf.zeros(stat_shape[3]))

    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift

    return output

