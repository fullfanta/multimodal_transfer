import tensorflow as tf
from network_common import *        

class StyleSubnet:
    def __init__(self, name='style_subnet'):
        self.name=name
        self.resized_image = None

    def inference(self, image, short_edge, reuse = False):
        with tf.variable_scope(self.name, reuse) as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(image)[0]
            height = tf.cast(tf.shape(image)[1], tf.float32)
            width = tf.cast(tf.shape(image)[2], tf.float32)

    
            # resize
            new_shorter_edge = short_edge
            height_smaller_than_width = tf.less_equal(height, width)
            new_height, new_width = tf.cond(
                height_smaller_than_width,
                lambda: (new_shorter_edge, tf.cast(width / height * tf.cast(new_shorter_edge, tf.float32), tf.int32)),
                lambda: (tf.cast(height / width * tf.cast(new_shorter_edge, tf.float32), tf.int32), new_shorter_edge))
            
            image = tf.image.resize_images(image, [new_height, new_width])
            self.resized_image = image

            # RGB block
            with tf.variable_scope('RGB-block', reuse) as scope:
                with tf.variable_scope('conv1', reuse) as scope:
                    W_conv = weight_variable([ 9, 9, 3, 32 ])
                    conv = conv2d(image, W_conv, stride = 1)
                    conv = inst_norm(conv)
                    conv1 = tf.nn.relu(conv)

                with tf.variable_scope('conv2', reuse) as scope:
                    W_conv = weight_variable([ 3, 3, 32, 64 ])
                    conv = conv2d(conv1, W_conv)
                    conv = inst_norm(conv)
                    conv2 = tf.nn.relu(conv)

                with tf.variable_scope('conv3', reuse) as scope:
                    W_conv = weight_variable([ 3, 3, 64, 128 ])
                    conv = conv2d(conv2, W_conv)
                    conv = inst_norm(conv)
                    conv3 = tf.nn.relu(conv)

                residual1 = residual(conv3, 'residual1', 128, reuse)
                residual2 = residual(residual1, 'residual2', 128, reuse)
                RGB_residual3 = residual(residual2, 'residual3', 128, reuse)

            # L block
            with tf.variable_scope('L-block', reuse) as scope:
                # luminance
                L = tf.image.rgb_to_grayscale(image)

                with tf.variable_scope('conv1', reuse) as scope:
                    W_conv = weight_variable([ 9, 9, 1, 32 ])
                    conv = conv2d(L, W_conv, stride = 1)
                    conv = inst_norm(conv)
                    conv1 = tf.nn.relu(conv)

                with tf.variable_scope('conv2', reuse) as scope:
                    W_conv = weight_variable([ 3, 3, 32, 64 ])
                    conv = conv2d(conv1, W_conv)
                    conv = inst_norm(conv)
                    conv2 = tf.nn.relu(conv)

                with tf.variable_scope('conv3', reuse) as scope:
                    W_conv = weight_variable([ 3, 3, 64, 128 ])
                    conv = conv2d(conv2, W_conv)
                    conv = inst_norm(conv)
                    conv3 = tf.nn.relu(conv)

                residual1 = residual(conv3, 'residual1', 128, reuse)
                residual2 = residual(residual1, 'residual2', 128, reuse)
                L_residual3 = residual(residual2, 'residual3', 128, reuse)

            # Conv block
            with tf.variable_scope('conv-block', reuse) as scope:
                combine = tf.concat([RGB_residual3, L_residual3], 3)

                residual1 = residual(combine, 'residual1', 256, reuse)
                residual2 = residual(residual1, 'residual2', 256, reuse)
                residual3 = residual(residual2, 'residual3', 256, reuse)

                with tf.variable_scope('resize_conv_3', reuse) as scope:
                    shape = tf.shape(conv2)
                    resize = tf.image.resize_images(residual3, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    W_conv = weight_variable([ 3, 3, 256, 128 ])
                    conv = conv2d(resize, W_conv, stride=1)
                    conv = tf.reshape(conv, [shape[0], shape[1], shape[2], 128])
                    conv = inst_norm(conv)
                    conv = tf.nn.relu(conv)

                with tf.variable_scope('resize_conv_2', reuse) as scope:
                    shape = tf.shape(conv1)
                    resize = tf.image.resize_images(conv, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    W_conv = weight_variable([ 3, 3, 128, 64 ])
                    conv = conv2d(resize, W_conv, stride=1)
                    conv = tf.reshape(conv, [shape[0], shape[1], shape[2], 64])
                    conv = inst_norm(conv)
                    conv = tf.nn.relu(conv)

                with tf.variable_scope('resize_conv_1', reuse) as scope:
                    shape = tf.shape(image)
                    resize = tf.image.resize_images(conv, [shape[1], shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    W_conv = weight_variable([ 3, 3, 64, 3 ])
                    conv = conv2d(resize, W_conv, stride=1)
                    conv = tf.nn.tanh(conv) * 255.0 + 255.0
                    conv = tf.div(conv, 2, name='output')

            return self.resized_image, conv
