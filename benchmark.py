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
        acc_3dp, acc_4dp, acc_5dp = sess.run([nn.acc_3dp, nn.acc_4dp, nn.acc_5dp], feed_dict=test_feed_dict)
        print('Test acc3dp:', acc_3dp, 'acc4dp:', acc_4dp, 'acc5dp:', acc_5dp)

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
    ID = 1
    test_filename = 'deep_impliedvol/lab_data_3_1/test_uniform.csv'
    feature_cols = [0, 1, 2]
    label_cols = [3]
    mini_batch_size = 100
    nn_layer_sizes = [64, 32]
    lr = 0
    seed = 1
    nb_epochs = 2


    LOGDIR = "run%s/" % str(ID)
    model_path = './run%s/checkpoints/' % str(ID)
   

    # test_accuracy(test_filename, feature_cols, label_cols, nn_layer_sizes, model_path)

    measure_speed(test_filename, feature_cols, label_cols, nn_layer_sizes, model_path)