import tensorflow as tf
from style_subnet import *
from enhance_subnet import *
from refine_subnet import *
from glob import glob
from vgg16 import vgg16
import cv2
import os
import numpy as np
import time
import datetime

FLAGS = tf.app.flags.FLAGS

def compute_gram(features):
    gram_list = []
    for feature in features:
        shape = tf.shape(feature)
        psi = tf.reshape(feature, [shape[0], shape[1] * shape[2], shape[3]])
        gram = tf.matmul(psi, psi, transpose_a = True)
        gram = tf.div(gram, tf.cast(shape[1] * shape[2] * shape[3], tf.float32))
        gram_list.append( gram )
    return gram_list

def train(argv=None):

    # compute gram losses from the hierarchy of style image before building networks
    # use gram list as constant
    input_style_image = cv2.imread(FLAGS.style_image)
    input_style_image = cv2.cvtColor(input_style_image, cv2.COLOR_BGR2RGB)

    print "compute gram matrix from style image"
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        vgg = vgg16('vgg16_weights.npz', sess)

        # 256 x 256
        style_image = tf.placeholder(tf.float32, [1, 256, 256, 3])
        _, style_features = vgg.get_features(style_image)
        style_gram_list = compute_gram(style_features)
        target_gram_256_list = sess.run(style_gram_list, feed_dict = {style_image : [cv2.resize(input_style_image, (256, 256))]})

        # 512 x 512
        style_image = tf.placeholder(tf.float32, [1, 512, 512, 3])
        _, style_features = vgg.get_features(style_image)
        style_gram_list = compute_gram(style_features)
        target_gram_512_list = sess.run(style_gram_list, feed_dict = {style_image : [cv2.resize(input_style_image, (512, 512))]})

        # 1024 x 1024
        style_image = tf.placeholder(tf.float32, [1, 1024, 1024, 3])
        _, style_features = vgg.get_features(style_image)
        style_gram_list = compute_gram(style_features)
        target_gram_1024_list = sess.run(style_gram_list, feed_dict = {style_image : [cv2.resize(input_style_image, (1024, 1024))]})

    hierarchical_weights = [float(w) for w in FLAGS.hierarchical_weights.split(',')]
    
    tf.reset_default_graph()
    # build multimodal transfer metnwork
    vgg = vgg16()
    input_image = tf.placeholder(tf.float32, [None, FLAGS.train_size, FLAGS.train_size, 3], name='input_image')
    short_edge_1 = tf.placeholder(tf.int32, shape=[], name='short_edge_1')
    short_edge_2 = tf.placeholder(tf.int32, shape=[], name='short_edge_2')
    short_edge_3 = tf.placeholder(tf.int32, shape=[], name='short_edge_3')

    input_content_features = []
    generated_content_features = []
    generated_style_features = []
    target_grams = [target_gram_256_list, target_gram_512_list, target_gram_1024_list]

    ## style subnet
    style_subnet = StyleSubnet('style_subnet')
    resized_input_image_1, generated_image_1 = style_subnet.inference(input_image, short_edge_1)
    generated_content_feature_1, generated_style_feature_1 = vgg.get_features(generated_image_1)
    input_content_feature_1, _ = vgg.get_features(resized_input_image_1)
    
    ## enhance subnet
    enhance_subnet = EnhanceSubnet('enhance_subnet')
    _, generated_image_2 = enhance_subnet.inference(generated_image_1, short_edge_2)
    generated_content_feature_2, generated_style_feature_2 = vgg.get_features(generated_image_2)
    input_content_feature_2, _ = vgg.get_features(input_image)

    ## refine subnet
    refine_subnet = RefineSubnet('refine_subnet')
    resized_input_image_3, generated_image_3 = refine_subnet.inference(generated_image_2, short_edge_3)
    generated_content_feature_3, generated_style_feature_3 = vgg.get_features(generated_image_3)
    input_content_feature_3, _ = vgg.get_features(input_image)
    
    input_content_features = [input_content_feature_1, input_content_feature_2, input_content_feature_3]
    generated_content_features = [generated_content_feature_1, generated_content_feature_2, generated_content_feature_3]
    generated_style_features = [generated_style_feature_1, generated_style_feature_2, generated_style_feature_3]

    # feature reconstrution loss
    feature_reconstruction_loss_list = []
    for i in range(0, len(input_content_features)):
        input_content_feature = input_content_features[i]
        generated_content_feature = generated_content_features[i]
        feature_shape = tf.shape(generated_content_feature)
        feature_size = tf.cast(feature_shape[1] * feature_shape[2] * feature_shape[3], dtype=tf.float32)
        feature_reconstruction_loss = hierarchical_weights[i] * FLAGS.content_weight * tf.reduce_sum(tf.squared_difference(generated_content_feature, input_content_feature)) / feature_size
        feature_reconstruction_loss_list.append( feature_reconstruction_loss )
    feature_reconstruction_loss = tf.add_n(feature_reconstruction_loss_list)

    # style reconstruction loss
    style_loss_list = []
    for i in range(0, len(generated_style_features)):
        generated_style_feature = generated_style_features[i]
        generated_style_gram_list = compute_gram(generated_style_feature)
        for j in range(0, len(generated_style_gram_list)): # num of feature maps
            shape = tf.shape(generated_style_gram_list[j])
            feature_size = tf.cast(shape[1] * shape[2], tf.float32)
            layer_style_loss = FLAGS.style_weight * tf.reduce_sum((generated_style_gram_list[j] - tf.constant(target_grams[i][j])) ** 2) / feature_size
            style_loss_list.append(hierarchical_weights[i] * layer_style_loss)
    style_reconstruction_loss = tf.add_n(style_loss_list)
    
    total_loss = feature_reconstruction_loss + style_reconstruction_loss
    
    # only updates stylization parameters
    train_vars_style = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='style_subnet')
    train_vars_enhance = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='enhance_subnet')
    train_vars_refine = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='refine_subnet')
    train_vars = train_vars_style + train_vars_enhance + train_vars_refine
    train_optimizer = tf.train.AdamOptimizer(0.001).minimize(total_loss, var_list = train_vars)

    tf.summary.scalar('feature reconstruction loss', feature_reconstruction_loss)
    tf.summary.scalar('style reconstruction loss', style_reconstruction_loss)
    tf.summary.image('input image', input_image, max_outputs=4)
    tf.summary.image('generated_image_1', generated_image_1, max_outputs=4)
    tf.summary.image('generated_image_2', generated_image_2, max_outputs=4)
    tf.summary.image('generated_image_3', generated_image_3, max_outputs=4)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver.
    saver = tf.train.Saver(train_vars)

    # get train images
    train_image_batch, num_train_images = get_train_images()
    iteration = num_train_images / FLAGS.batch_size

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess: 
        # initialize the variables
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        
        # load weight again because variables are all initialized
        vgg.load_weights('vgg16_weights.npz', sess) 

        # write graph definition
        tf.train.write_graph(sess.graph_def, FLAGS.summary_dir, '%s_graph_def.pb' % (FLAGS.style_image.split('/')[-1].split('.')[0]))

        # summary
        summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        short_edges = [int(w) for w in FLAGS.hierarchical_short_edges.split(',')]

        count = 0
        for i in range(0, FLAGS.epoch):
            for j in range(0, iteration):
                
                # get train image batch
                train_images = sess.run(train_image_batch)
                
                # train network with batch
                sess.run(train_optimizer, feed_dict = {input_image : train_images, short_edge_1 : short_edges[0], short_edge_2 : short_edges[1], short_edge_3 : short_edges[2]})

                # write summary 
                if count % 10 == 0:
                    _, output_summary, output_f_loss, output_s_loss = sess.run([train_optimizer, summary_op, feature_reconstruction_loss, style_reconstruction_loss], feed_dict = {input_image : train_images, short_edge_1 : short_edges[0], short_edge_2 : short_edges[1], short_edge_3 : short_edges[2]})
                
                    summary_writer.add_summary(output_summary, count)

                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                    print '%s, epoch[%d] iter[%d] : feature loss - %f, style loss - %f' % (st, i, j, output_f_loss, output_s_loss)
                
                
                # save
                if count % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.summary_dir, '%s.ckpt' % (FLAGS.style_image.split('/')[-1].split('.')[0]))
                    saver.save(sess, checkpoint_path, global_step=count, write_meta_graph = False)

                count += 1


def get_train_images():
    image_list = glob(FLAGS.train_path + '/*.jpg')
    print len(image_list)
    
    train_image_name = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    train_input_queue = tf.train.slice_input_producer([train_image_name], shuffle = True)
    
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = tf.cast(train_image, tf.float32)
    train_image = tf.image.resize_images(train_image, [FLAGS.train_size, FLAGS.train_size])
    train_image.set_shape([FLAGS.train_size, FLAGS.train_size, 3])

    min_after_dequeue = 100
    capacity = min_after_dequeue + 4 * FLAGS.batch_size
    train_image = tf.train.shuffle_batch(
        [train_image],
        batch_size=FLAGS.batch_size
        ,num_threads=4
        , capacity=capacity
        , min_after_dequeue=min_after_dequeue
    )

    return train_image, len(image_list)


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('batch_size', 2, """The batch size to use.""")
    tf.app.flags.DEFINE_float('content_weight', 1, """weight for content reconstruction loss.""")
    tf.app.flags.DEFINE_float('style_weight', 5, """weight for style reconstruction loss.""")
    tf.app.flags.DEFINE_string('hierarchical_weights', '1,1,1', """weigts for hierarchical stylization loss.""")
    tf.app.flags.DEFINE_string('hierarchical_short_edges', '256,512,512', """short edges for hierarchy training""")
    tf.app.flags.DEFINE_string('summary_dir', './summary', """summary directory.""")
    tf.app.flags.DEFINE_string('style_image', './style_images/starry_night.jpg', """target style image""")
    tf.app.flags.DEFINE_integer('train_size', 512, """image width and height""")
    tf.app.flags.DEFINE_string('train_path', './data/train2014', """path which contains train images""")
    tf.app.flags.DEFINE_integer('epoch', 5, """epoch""")
    

    # clear summary directory
    log_files = glob(FLAGS.summary_dir + '/events*')
    for f in log_files:
        os.remove(f)

    tf.app.run(main=train)
