# LEARN BLACKSCHOLES IMPLIED VOLATILITY WITH A NEURAL NETWORK
# ==============================================================================

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from collections import namedtuple

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)


def fc_layer(input, dim_in, dim_out, name='fc_layer'):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=2.0/dim_in), name='W')
        B = tf.Variable(tf.zeros([dim_out]), name='B')
        nonlinearity = tf.nn.relu(tf.matmul(input, W) + B)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("nonlinearity", nonlinearity)
        return nonlinearity


def rank1_ff_nn(nb_features, nn_layer_sizes, nb_labels):

    nn = namedtuple('nn', ['inputs', 'labels', 'pred', 'loss'])

    tf.reset_default_graph()

    # Placeholders for labeled pair of training data.
    nn.inputs = tf.placeholder(tf.float32, [None, nb_features], name='inputs')
    nn.labels = tf.placeholder(tf.float32, [None, nb_labels], name='labels')

    # Build the computational graph consisting of a feed forward NN. 
    nb_hidden_layers = len(nn_layer_sizes)

    layers = []

    layers.append(fc_layer(nn.inputs, nb_features, nn_layer_sizes[0], 'fc_hidden0'))

    for i in range(nb_hidden_layers - 1):

        layers.append(fc_layer(layers[i], nn_layer_sizes[i], 
                      nn_layer_sizes[i+1], 'fc_hidden%s' % str(i+1)))

    nn.pred = fc_layer(layers[-1], nn_layer_sizes[-1], nb_labels, 'pred')

    layers.append(nn.pred)

    # Define the loss function.
    with tf.name_scope('loss'):
        nn.loss = tf.reduce_sum(tf.square(nn.pred-nn.labels))
        tf.summary.scalar('loss', nn.loss)

    # Define accuracy to be % of predictions within certain delta of labels.
    with tf.name_scope('accuracy'):

        # Accuracy to 1E-3
        close_prediction_3dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-3)
        nn.acc_3dp = tf.reduce_mean(tf.cast(close_prediction_3dp, tf.float32))
        tf.summary.scalar('accuracy_3dp', nn.acc_3dp)

        # Accuracy to 1E-4
        close_prediction_4dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-4)
        nn.acc_4dp = tf.reduce_mean(tf.cast(close_prediction_4dp, tf.float32))
        tf.summary.scalar('accuracy_4dp', nn.acc_4dp)

        # Accuracy to 1E-5
        close_prediction_5dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-5)
        nn.acc_5dp = tf.reduce_mean(tf.cast(close_prediction_5dp, tf.float32))
        tf.summary.scalar('accuracy_5dp', nn.acc_5dp)

    return nn


def import_labeled_csv_data(filename, feature_cols, label_cols):

    data = namedtuple('data', ['features', 'labels', 'nb_features', 
                      'nb_labels', 'nb_samples'])

    data.features = pd.read_csv(filename, skiprows=0, usecols=feature_cols).values

    data.labels = pd.read_csv(filename, skiprows=0, usecols=label_cols).values

    data.nb_samples, data.nb_features = data.features.shape

    data.nb_labels = data.labels.shape[1]

    return data


def train(train_set_csv, valid_set_csv, mini_batch_size, nn_layer_sizes, lr, seed, nb_epochs):

    # Initialization.
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed+1)

    # Read training and validation data named tuples into memory.
    train_set = import_labeled_csv_data(train_set_csv, [0], [1])
    val_set = import_labeled_csv_data(valid_set_csv, [0], [1]) 

    # Build the computational graph of a feed-forward NN.
    nn = rank1_ff_nn(train_set.nb_features, nn_layer_sizes, train_set.nb_labels)

    # Build the training op.
    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(lr).minimize(nn.loss)

    # Print neural network configuration.
    net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
    hyp_param_settings = net_config + ",lr_%.0E" % (lr)
    print("Neural network built with hyperparameter settings:", hyp_param_settings)  

    # Collect all summary ops in one op.
    summary = tf.summary.merge_all()

    # Build the validation set dictionary.
    val_feed_dict = { nn.inputs : val_set.features,
                      nn.labels : val_set.labels}

    # Run session through the computational graph.
    with tf.Session() as sess:

        # Init.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(LOGDIR + hyp_param_settings, graph=sess.graph)
    
        # Perform training cycles.
        for epoch in range(nb_epochs):

            # Compute how many minibatches to run through.
            nb_mini_batches = int(train_set.nb_samples/mini_batch_size)

            # Do random shuffling of the indices of training samples.
            shuffled_indices = np.random.permutation(train_set.nb_samples)

            # Running through individual minibatches and doing backprop.
            for i in range(nb_mini_batches):

                mini_batch_indices = shuffled_indices[i:i + mini_batch_size]

                train_feed_dict = { nn.inputs : train_set.features[mini_batch_indices, :],
                                    nn.labels: train_set.labels[mini_batch_indices, :]
                                   }

                # Run training step (which includes backpropagation).
                sess.run([train_step], feed_dict=train_feed_dict)

         
            
            # Writing results on validation set to disk.
            validation_summary = sess.run(summary, feed_dict = val_feed_dict)
            writer.add_summary(validation_summary, epoch)

            # Printing accuracies at different levels to see training of NN.
            loss, acc_3dp, acc_4dp, acc_5dp = sess.run([nn.loss, nn.acc_3dp, nn.acc_4dp, nn.acc_5dp], feed_dict= val_feed_dict)
            print('Epoch: ', epoch, 'loss:', loss, 'acc3dp: ', acc_3dp, 'acc4dp: ', acc_4dp, 'acc5dp: ', acc_5dp)

            # Save checkpoint files for reuse later.
            saver.save(sess, save_path=model_path, global_step=epoch)

            # Stop performing training cycles if network is accurate enough.
            if acc_4dp > 0.99:

                break

        # Saving final model.
        save_path = saver.save(sess, model_path + 'final_model')

        print("Model saved in file: %s" % save_path)

    print('Run `tensorboard --logdir=%s/%s` to see the results.' % (CURRENT_PATH, LOGDIR))


def test_acc(test_set_csv, nn_layer_sizes, model_path):

    tf.reset_default_graph()

    # Read test data named tuple into memory. 
    test_set = import_labeled_csv_data(test_set_csv, [0], [1])

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


def time_benchmark(test_set_csv, nn_layer_sizes, model_path):

    nb_tries = 100

    # Initialization.
    tf.reset_default_graph()
    test_set = import_labeled_csv_data(test_set_csv, [0], [1])
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

            print(duration)

            avg_time += duration

        avg_time = avg_time/nb_tries

    print('%i tries, %i data points, avg time: %f' % (nb_tries, test_set.nb_samples, avg_time))
 

if __name__ == '__main__':

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    # Configuration.
    ID = 2
    train_set_csv = 'lab_data_1_1/train_uniform.csv'
    valid_set_csv = 'lab_data_1_1/validation_uniform.csv'
    test_set_csv = 'lab_data_1_1/test_uniform.csv'
    mini_batch_size = 100
    nn_layer_sizes = [64, 32]
    lr = 0.0001
    seed = 0
    nb_epochs = 10000

    LOGDIR = "run%s/" % str(ID)
    model_path = './run%s/checkpoints/' % str(ID)

    # train(train_set_csv, valid_set_csv, mini_batch_size, nn_layer_sizes, lr, seed, nb_epochs)

    # test_acc(test_set_csv, nn_layer_sizes, model_path)

    time_benchmark(test_set_csv, nn_layer_sizes, model_path)

