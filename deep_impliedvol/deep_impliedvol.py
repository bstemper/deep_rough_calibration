# LEARN BLACKSCHOLES IMPLIED VOLATILITY WITH A NEURAL NETWORK
# ==============================================================================

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import namedtuple

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)


def fc_layer(input, dim_in, dim_out, name='fc_layer'):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1), name='W')
        B = tf.Variable(tf.ones([dim_out])/10, name='B')
        nonlinearity = tf.nn.relu(tf.matmul(input, W) + B)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("nonlinearity", nonlinearity)
        return nonlinearity


def rank1_ff_nn(nb_features, nn_layer_sizes, nb_labels):

    nn = namedtuple('nn', ['input', 'labels', 'pred', 'loss'])

    tf.reset_default_graph()

    # Placeholders for labeled pair of training data.
    nn.input = tf.placeholder(tf.float32, [None, nb_features], name='input')
    nn.labels = tf.placeholder(tf.float32, [None, nb_labels], name='labels')

    # Build the computational graph consisting of a feed forward NN. 
    nb_hidden_layers = len(nn_layer_sizes)

    layers = []

    layers.append(fc_layer(nn.input, nb_features, nn_layer_sizes[0], 'fc_hidden0'))

    for i in range(nb_hidden_layers - 1):

        layers.append(fc_layer(layers[i], nn_layer_sizes[i], 
                      nn_layer_sizes[i+1], 'fc_hidden%s' % str(i+1)))

    nn.pred = fc_layer(layers[-1], nn_layer_sizes[-1], nb_labels, 'pred')

    layers.append(nn.pred)

    # Define the loss function.
    with tf.name_scope('loss'):
        nn.loss = tf.reduce_sum(tf.square(nn.pred-nn.labels))
        tf.summary.scalar('loss', nn.loss)

    return nn


def import_labeled_csv_data(filename, feature_cols, label_cols):

    data = namedtuple('data', ['features', 'labels', 'nb_features', 
                      'nb_labels', 'nb_samples'])

    data.features = pd.read_csv(filename, skiprows=0, usecols=feature_cols).values

    data.labels = pd.read_csv(filename, skiprows=0, usecols=label_cols).values

    data.nb_samples, data.nb_features = data.features.shape

    data.nb_labels = data.labels.shape[1]

    return data


def train(train_set_csv, val_set_csv, mini_batch_size, nn_layer_sizes, lr, seed, nb_epochs):

    # Set random seed for reproducibility and comparability.
    tf.set_random_seed(seed)

    # Read training and validation data named tuples into memory.
    train_set = import_labeled_csv_data(train_set_csv, [0], [1])
    validation_set = import_labeled_csv_data(val_set_csv, [0], [1])

    # Build the computational graph of a feed-forward NN.
    nn = rank1_ff_nn(train_set.nb_features, nn_layer_sizes, train_set.nb_labels)

    # Build the training op.
    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(lr).minimize(nn.loss)

    # Define accuracy to be % of predictions within certain delta of labels.
    with tf.name_scope('accuracy'):

        # Accuracy to 1E-3
        close_prediction_3dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-3)
        accuracy_3dp = tf.reduce_mean(tf.cast(close_prediction_3dp, tf.float32))
        tf.summary.scalar('accuracy_3dp', accuracy_3dp)

        # Accuracy to 1E-4
        close_prediction_4dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-4)
        accuracy_4dp = tf.reduce_mean(tf.cast(close_prediction_4dp, tf.float32))
        tf.summary.scalar('accuracy_4dp', accuracy_4dp)

        # Accuracy to 1E-5
        close_prediction_5dp = tf.less_equal(tf.abs(nn.pred-nn.labels), 1E-5)
        accuracy_5dp = tf.reduce_mean(tf.cast(close_prediction_5dp, tf.float32))
        tf.summary.scalar('accuracy_5dp', accuracy_5dp)

    summary = tf.summary.merge_all()

    # Run session through the computational graph.
    with tf.Session() as sess:

        # Initialize all variables in the graph.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Create writers for train and validation data.
        hparam = " %s, lr_%.0E" % (nn_layer_sizes, lr)
        writer = tf.summary.FileWriter(LOGDIR + hparam, graph=sess.graph)

        # Perform training cycles.
        for epoch in range(nb_epochs):

            # Compute how many minibatches to run through.
            nb_mini_batches = int(train_set.nb_samples/mini_batch_size)

            # Do random shuffling of the indices of training samples.
            shuffled_indices = np.random.permutation(train_set.nb_samples)

            # Running through individual minibatches and doing backprop
            for i in range(nb_mini_batches):

                mini_batch_indices = shuffled_indices[i:i + mini_batch_size]

                train_feed_dict = { nn.input : train_set.features[mini_batch_indices, :],
                                    nn.labels: train_set.labels[mini_batch_indices, :]
                                   }

                # Run training step (which includes backpropagation).
                sess.run([train_step], feed_dict=train_feed_dict)

         
            # Writing tensorboard summaries to disk.
            val_feed_dict = { nn.input : validation_set.features,
                              nn.labels : validation_set.labels
                            }

            validation_summary = sess.run(summary, feed_dict = val_feed_dict)

            writer.add_summary(validation_summary, epoch)

            acc_3dp, acc_4dp, acc_5dp = sess.run([accuracy_3dp, accuracy_4dp, accuracy_5dp], feed_dict= val_feed_dict)

            print('Epoch: ', epoch, 'acc3dp: ', acc_3dp, 'acc4dp: ', acc_4dp, 'acc5dp: ', acc_5dp)

            # Stop performing training cycles if network is accurate enough.
            if acc_3dp > 0.99:

                break


    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)


   

if __name__ == '__main__':


    # Configuration.
    train_set_csv = '1d_labeled_data/train_uniform_1d.csv'
    val_set_csv = '1d_labeled_data/validation_uniform_1d.csv'
    mini_batch_size = 100
    nn_layer_sizes = [64, 32]
    lr = 0.0005
    seed = 0
    nb_epochs = 10000

    LOGDIR = "/tmp/test/"

    train(train_set_csv, val_set_csv, mini_batch_size, nn_layer_sizes, lr, seed, nb_epochs)
