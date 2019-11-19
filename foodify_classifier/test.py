import tensorflow as tf, sys, argparse

def main(args):
    image_data = tf.gfile.FastGFile(args.image_path, 'rb').read()
    label_lines = [line.rstrip() for line
        in tf.gfile.GFile(args.labels)]
    with tf.gfile.FastGFile(args.graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, 
        {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str,
                         help='The image path of the testing image.')
    parser.add_argument('--graph', type=str,
                        help='The path of trained graph.',
                        default='./train_files/output_graph.pb')
    parser.add_argument('--labels', type=str,
                        help='The path of trained labels.',
                        default='./train_files/output_labels.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
