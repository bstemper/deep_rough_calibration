import tensorflow as tf
from collections import namedtuple


def fc_layer(input, dim_in, dim_out, random_seed, name='fc_layer'):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=2.0/dim_in, seed=random_seed), name='W')
        B = tf.Variable(tf.zeros([dim_out]), name='B')
        nonlinearity = tf.nn.relu(tf.matmul(input, W) + B)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("nonlinearity", nonlinearity)
        return nonlinearity


def rank1_ff_nn(nb_features, nn_layer_sizes, nb_labels, random_seed):

    nn = namedtuple('nn', ['inputs', 'labels', 'pred', 'loss'])

    tf.reset_default_graph()

    # Placeholders for labeled pair of training data.
    nn.inputs = tf.placeholder(tf.float32, [None, nb_features], name='inputs')
    nn.labels = tf.placeholder(tf.float32, [None, nb_labels], name='labels')

    # Build the computational graph consisting of a feed forward NN. 
    nb_hidden_layers = len(nn_layer_sizes)

    layers = []

    layers.append(fc_layer(nn.inputs, nb_features, nn_layer_sizes[0], random_seed,
                           'fc_hidden0'))

    for i in range(nb_hidden_layers - 1):

        layers.append(fc_layer(layers[i], nn_layer_sizes[i], 
                      nn_layer_sizes[i+1], random_seed, 'fc_hidden%s' % str(i+1)))

    nn.pred = fc_layer(layers[-1], nn_layer_sizes[-1], nb_labels, random_seed, 'pred')

    layers.append(nn.pred)

    # Define the loss function.
    with tf.name_scope('loss'):
        nn.loss = tf.reduce_sum(tf.square(nn.pred-nn.labels))
        tf.summary.scalar('loss', nn.loss)

    # Let accuracy be % of preds with relative error below a certain threshold.
    with tf.name_scope('accuracy'):

        relative_error = tf.abs(nn.pred-nn.labels)/nn.labels

        # Relative error less than 2%
        close_prediction_2pc = tf.less_equal(relative_error, 0.02)
        nn.acc_2pc = tf.reduce_mean(tf.cast(close_prediction_2pc, tf.float32))
        tf.summary.scalar('accuracy_2pc', nn.acc_2pc)

        # Relative error less than 1%
        close_prediction_1pc = tf.less_equal(relative_error, 0.01)
        nn.acc_1pc = tf.reduce_mean(tf.cast(close_prediction_1pc, tf.float32))
        tf.summary.scalar('accuracy_1pc', nn.acc_1pc)

    return nn

    