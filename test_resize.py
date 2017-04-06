import tensorflow as tf
import argparse
import cv2



def main():
    # load image
    input_image = cv2.imread(args.input_image, cv2.CV_LOAD_IMAGE_COLOR)
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    print 'input image - ', args.input_image, input_image.shape

    input_tensor = tf.placeholder(tf.float32, [1, input_image.shape[0], input_image.shape[1], 3])


    shape = tf.shape(input_tensor)
    #shape = input_tensor.get_shape().as_list()
    height = tf.cast(shape[1], tf.float32)
    width = tf.cast(shape[2], tf.float32)

    new_shorter_edge = 256
    height_smaller_than_width = tf.less_equal(height, width)
    new_shorter_edge = tf.constant(256)
    (new_height, new_width) = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, tf.cast(width / height * tf.cast(new_shorter_edge, tf.float32), tf.int32)),
        lambda: (tf.cast(height / width * tf.cast(new_shorter_edge, tf.float32), tf.int32), new_shorter_edge))

    output_tensor = tf.image.resize_images(input_tensor, [new_height, new_width])
    
    # generate
    with tf.Session() as sess:
        result = sess.run(output_tensor, feed_dict = {input_tensor : [input_image]})
        result = result[0]
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
 
        if args.output_image is None:
            idx = args.input_image.rfind('.')
            args.output_image = args.input_image[:idx] + '_output.jpg'
        cv2.imwrite(args.output_image, result)
        print 'output image - ', args.output_image


if __name__=='__main__':
    parser = argparse.ArgumentParser('Stylizer')
    parser.add_argument('--model', type=str, default='models/starry_night.pb')
    parser.add_argument('--input_image', type=str, default='./test_images/Aaron_Eckhart_0001.jpg')
    parser.add_argument('--output_image', type=str)
    parser.add_argument('--resize_ratio', type=int, default=1)
    
    args = parser.parse_args()  

    main()
