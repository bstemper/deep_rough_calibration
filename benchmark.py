import os
import sys
import tensorflow as tf
import time
from helpers import import_labeled_csv_data
from neural_network import rank1_ff_nn


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)

def test_accuracy(filename, feature_cols, label_cols, nn_layer_sizes, model_path):

    tf.reset_default_graph()

    # Read test data named tuple into memory. 
    test_set = import_labeled_csv_data(filename, feature_cols, label_cols)

    # Build the computational graph of a feed-forward NN.
    nn = rank1_ff_nn(test_set.nb_features, nn_layer_sizes, test_set.nb_labels)

    # Initialization.
    saver = tf.train.Saver()
    test_feed_dict = {nn.inputs : test_set.features,
                      nn.labels : test_set.labels}

    # Run session through the computational graph.
    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        # Printing accuracies at different levels to see quality of NN training.
        loss, acc_2pc, acc_1pc = sess.run([nn.loss, nn.acc_2pc, nn.acc_1pc], feed_dict=test_feed_dict)
        print('Test acc2pc:', acc_2pc, 'acc1pc:', acc_1pc)

def measure_speed(filename, feature_cols, label_cols, nn_layer_sizes, model_path):

    nb_tries = 100

    # Initialization.
    tf.reset_default_graph()
    test_set = import_labeled_csv_data(filename, feature_cols, label_cols)
    nn = rank1_ff_nn(test_set.nb_features, nn_layer_sizes, test_set.nb_labels)
    saver = tf.train.Saver()
    test_feed_dict = {nn.inputs : test_set.features,
                      nn.labels : test_set.labels}

    # Run session through the computational graph.
    with tf.Session() as sess:

        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        avg_time = 0

        for _ in range(nb_tries):

            start_time = time.time()

            sess.run([nn.pred], feed_dict=test_feed_dict)

            duration = time.time() - start_time

            avg_time += duration

        avg_time = avg_time/nb_tries

    print('%i tries, %i data points, avg time: %f' % (nb_tries, test_set.nb_samples, avg_time))

if __name__ == '__main__':

    # Configuration.
    test_filename = 'deep_impliedvol/labeled_data_toy/test_uniform.csv'
    feature_cols = [0]
    label_cols = [1]
    nn_layer_sizes = [64, 32]
    lr = 5E-05

    net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
    hyp_param_settings = net_config + ",lr_%.5E" % (lr)

    model_path = hyp_param_settings + '/'

    print(model_path)

    test_accuracy(test_filename, feature_cols, label_cols, nn_layer_sizes, model_path)

    measure_speed(test_filename, feature_cols, label_cols, nn_layer_sizes, model_path)

    