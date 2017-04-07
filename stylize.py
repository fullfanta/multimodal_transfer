import tensorflow as tf
import argparse
import cv2


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                name="prefix", 
                op_dict=None, 
                producer_op_list=None
            )
    return graph


def main():
    # load image
    input_image = cv2.imread(args.input_image, cv2.CV_LOAD_IMAGE_COLOR)
    #input_image = cv2.resize(input_image, (input_image.shape[1] / args.resize_ratio, input_image.shape[0] / args.resize_ratio))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    print 'input image - ', args.input_image, input_image.shape

    input_tensor = tf.placeholder(tf.float32, [1, input_image.shape[0], input_image.shape[1], 3])
    short_edge_1 = tf.placeholder(tf.int32, [])
    short_edge_2 = tf.placeholder(tf.int32, [])
    short_edge_3 = tf.placeholder(tf.int32, [])

    '''
    graph = load_graph(args.model)
    for op in graph.get_operations():
       print op.name 
    '''

    # load model
    with tf.gfile.GFile(args.model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            generated_image_1, generated_image_2, generated_image_3, = tf.import_graph_def(
                    graph_def, 
                    input_map={'input_image' : input_tensor, 'short_edge_1' : short_edge_1, 'short_edge_2' : short_edge_2, 'short_edge_3' : short_edge_3}, 
                    return_elements=['style_subnet/conv-block/resize_conv_1/output:0', 'enhance_subnet/resize_conv_1/output:0', 'refine_subnet/resize_conv_1/output:0'], 
                    name=None, 
                    op_dict=None, 
                    producer_op_list=None
                )

    short_edges = [int(e) for e in args.hierarchical_short_edges.split(',')]

    # generate
    with tf.Session() as sess:
        result_1, result_2, result_3 = sess.run([generated_image_1, generated_image_2, generated_image_3], feed_dict = {input_tensor : [input_image], short_edge_1 : short_edges[0], short_edge_2 : short_edges[1], short_edge_3 : short_edges[2]})
        result_1 = cv2.cvtColor(result_1[0], cv2.COLOR_BGR2RGB)
        result_2 = cv2.cvtColor(result_2[0], cv2.COLOR_BGR2RGB)
        result_3 = cv2.cvtColor(result_3[0], cv2.COLOR_BGR2RGB)
 
        idx = args.input_image.rfind('.')
        output_name_1 = args.input_image[:idx] + '_output_1.jpg'
        output_name_2 = args.input_image[:idx] + '_output_2.jpg'
        output_name_3 = args.input_image[:idx] + '_output_3.jpg'

        cv2.imwrite(output_name_1, result_1)
        cv2.imwrite(output_name_2, result_2)
        cv2.imwrite(output_name_3, result_3)
        print 'output image - ', output_name_1, output_name_2, output_name_3


if __name__=='__main__':
    parser = argparse.ArgumentParser('Stylizer')
    parser.add_argument('--model', type=str, default='models/starry_night.pb')
    parser.add_argument('--input_image', type=str, default='./test_images/Aaron_Eckhart_0001.jpg')
    parser.add_argument('--hierarchical_short_edges', type=str, default='256,512,1024')
   

    args = parser.parse_args()  

    main()
