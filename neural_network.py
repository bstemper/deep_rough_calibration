import tensorflow as tf
from collections import namedtuple


def fc_layer(input, dim_in, dim_out, random_seed, pkeep, name='fc_layer'):
    """
    Definition of a fully connected layer.
    
    Arguments:
    ----------
        input: tensor, shape=[None, dim_in].
            Input features or previous layer as input to this layer.
        dim_in: int.
            Number of features or neurons in previous layer.
        dim_out: int.
            Number of neurons of this layer.
        random_seed: int.
            PRNG setting used for weight initializiation.
        pkeep: float, has to be within (0,1).
            Probability of keeping neurons in dropout.
        name: string.
            Name of the layer (used for Tensorboard).

    Returns:
    --------
        nonlinearity_dropout: tensor, shape=[dim_out,].
            Output of layer run through dropout procedure.
    """

    with tf.name_scope(name):

        # Define fully-connected layer logic in tensorflow.
        # Weight initialization for ReLUs as per He. et al
        W = tf.Variable(tf.truncated_normal([dim_in, dim_out], 
                        stddev=2.0/dim_in, seed=random_seed), name='weights')

        B = tf.Variable(tf.zeros([dim_out]), name='bias')

        nonlinearity = tf.nn.relu(tf.matmul(input, W) + B, name='nonlinearity')

        nonlinearity_dropout = tf.nn.dropout(nonlinearity, pkeep)

        # Collecting summaries for tensorboard.
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("nonlinearity", nonlinearity)
        tf.summary.histogram("nonlinearity_dropout", nonlinearity_dropout)

        return nonlinearity_dropout


def fully_connected_nn(nb_features, layer_sizes, nb_labels, random_seed):
    """
    Builds a fully connected neural network with specified topology and returns
    in a named tuple the crucial ops to be used in training and
    benchmarking functions.

    Arguments:
    ----------
        nb_features: int.
            Number of features in labeled data.
        layer_sizes: array-like, shape=[, # layers].
            Number of units per layer, e.g. [32,16,8,4].
        nb_labels: int.
            Number of labels in labeled data.
        random_seed: int.
            PRNG setting used for weight initializiation.

    Returns:
    --------
        nn: named tuple.
            nn.inputs: tf.placeholder(tf.float32, [None, nb_features]).
                Tensorflow input placeholder for neural network.
            nn.labels: tf.placeholder(tf.float32, [None, nb_labels]).
                Tensorflow placeholder for labels.
            nn.pkeep: tf.placeholder(tf.float32).
                Regularisation via Dropout: Probability of keeping nodes.
            nn.predictions: tf op(tf.float32, [None, nb_labels]).
                Final/output layer of neural network.
            nn.loss: tf op (tf.float32)
                Loss op in computational graph computing loss function.
            nn.err_2pc: tf op (tf.float32)
                Percentage of predictions with relative error of more than 2%.
            nn.err_1pc: tf op (tf.float32)
                Percentage of predictions with relative error of more than 1%.

    """

    ## INITIALIZATION

    tf.reset_default_graph()

    # Creating a class of named tuples collecting neural network ops.
    NeuralNetwork = namedtuple('nn', 'inputs, labels, pkeep, predictions, \
                                loss, err_2pc, err_1pc')

    # Placeholders for labeled pair of training data.
    inputs = tf.placeholder(tf.float32, [None, nb_features], name='inputs')
    labels = tf.placeholder(tf.float32, [None, nb_labels], name='labels')

    # Regularisation via Dropout: Probability of keeping nodes.
    pkeep = tf.placeholder(tf.float32)

    ## CONSTRUCTION OF COMPUTATIONAL GRAPH OF FULLY CONNECTED NN

    nb_hidden_layers = len(layer_sizes)
    layers = []

    # Dealing with special case of first hidden layer.
    first_layer = fc_layer(inputs, nb_features, layer_sizes[0], random_seed,
                           pkeep, 'fc_hidden_0')

    layers.append(first_layer)

    # Dealing with hidden layers between first and final prediction layer.
    for i in range(nb_hidden_layers - 1):

        hidden_layer = fc_layer(layers[i], layer_sizes[i], layer_sizes[i+1], 
                                random_seed, pkeep, 'fc_hidden_%s' % str(i+1))

        layers.append(hidden_layer)

    # Dealing with final prediction layer.
    prediction_layer = fc_layer(layers[-1], layer_sizes[-1], nb_labels, 
                                random_seed, 1, 'predictions')

    layers.append(prediction_layer)

    ## ADDING LOSS & ACCURACY/ERROR TO COMPUTATIONAL GRAPH

    # Define the loss function.
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(prediction_layer-labels))
        tf.summary.scalar('loss', loss)

    # Define accuracy = % of predictions with RE < certain threshold.
    with tf.name_scope('accuracy'):

        # Define the relative error as a metric of accuracy for predictions.
        relative_error = tf.abs(prediction_layer-labels)/labels

        # Relative error less than 2%
        close_prediction_2pc = tf.greater(relative_error, 0.02)
        err_2pc = tf.reduce_mean(tf.cast(close_prediction_2pc, tf.float32))
        tf.summary.scalar('error_2pc', err_2pc)

        # Relative error less than 1%
        close_prediction_1pc = tf.greater(relative_error, 0.01)
        err_1pc = tf.reduce_mean(tf.cast(close_prediction_1pc, tf.float32))
        tf.summary.scalar('error_1pc', err_1pc)

    ## COLLECTION OPS AND INFOS OF NN IN NAMED TUPLE

    nn = NeuralNetwork(inputs = inputs,
                       labels = labels,
                       pkeep  = pkeep,
                       predictions = prediction_layer,
                       loss = loss,
                       err_2pc = err_2pc,
                       err_1pc = err_1pc)

    return nn

    