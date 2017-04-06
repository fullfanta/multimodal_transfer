########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

# downloaded from https://www.cs.toronto.edu/~frossard/post/vgg16/


import tensorflow as tf
import numpy as np
import cv2


class vgg16:
    def __init__(self, weights=None, sess=None):
        self.parameters = []
        self.initialize_weights()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def initialize_weights(self):
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]   

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]    

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:      
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]


        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.parameters += [kernel, biases]

    def load_weights(self, weight_file, sess):
        print 'load weights for vgg16 from - ', weight_file
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            # only loads convolution weights
            if i < len(self.parameters):
                sess.run(self.parameters[i].assign(weights[k]))


    def convlayers(self, input, reuse = True):
        # zero-mean input
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        images = input - mean

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            if reuse:
                scope.reuse_variables()    
            kernel = self.parameters[0]
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[1]
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out)

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            if reuse:
                scope.reuse_variables()    
            kernel = self.parameters[2]
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[3]
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = self.parameters[4]
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[5]
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out)

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            if reuse:
                scope.reuse_variables()            
            kernel = self.parameters[6]
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[7]
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[8]
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[9]
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out)

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[10]
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[11]
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out)

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[12]
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[13]
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[14]
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[15]
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out)

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[16]
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[17]
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out)

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[18]
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[19]
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[20]
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[21]
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out)

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[22]
            conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[23]
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out)

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            if reuse:
                scope.reuse_variables()
            kernel = self.parameters[24]
            conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self.parameters[25]
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out)

        # pool5
        pool5 = tf.nn.max_pool(conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        return [conv1_1, conv2_1, conv3_1, conv4_1, conv4_2]

    




    def get_features(self, image):
        layers = self.convlayers(image, reuse=True)
        return layers[2], [layers[0], layers[1], layers[2], layers[3]]
        


if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    

    img1 = cv2.imread('laska.png')
    img1 = cv2.resize(img1, (224, 224))

    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]
