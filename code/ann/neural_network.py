# ------------------- Neural network construction ------------------------- #

import tensorflow as tf
from math import sqrt
from collections import namedtuple

def dense_relu(inputs, units, name='dense_relu'):
    """
    Definition of a fully-connected layer with ReLU activations.

    Arguments:
    ----------
        inputs: tensor, shape=[None, dim_in].
            Features or previous layer as input to this dense layer.
        units: int.
            Number of units in this layer.
        name: string.
            Name of the layer (used for Tensorboard).

    Returns:
    --------
        dense_layer: tensor, shape=[None, units]
    """

    # Specify the number of units/features incoming.
    dim_in = inputs.get_shape().as_list()[1]

    with tf.variable_scope(name):

        # Weight initialization optimized for ReLUs a la He et al. (2015).
        kernel_init= tf.random_normal_initializer(stddev=sqrt(2.0/dim_in))

        # Functional interface for dense layer.
        dense_layer = tf.layers.dense(inputs, units, 
                                      activation=tf.nn.relu, 
                                      kernel_initializer=kernel_init) 

    return dense_layer
                                            

def dense_relu_bn_drop(inputs, units, training_phase, pkeep, 
                       name='dense_relu_bn_drop'):
    """
    Definition of a fully-connected layer: DENSE > BN > RELU > Dropout.
    
    Arguments:
    ----------
        inputs: tensor, shape=[None, dim_in].
            Input features or previous layer as input to this layer.
        units: int.
            Number of units of this layer.
        training_phase: boolean.
            For batchnorm: If training, set True. If test, set False.
        pkeep: float, in (0,1).
            For dropout: Probability of keeping units during droupout.
        name: string.
            Name of the layer (used for Tensorboard).

    Returns:
    --------
        output: tensor, shape=[None, units].
            Output of layer run through dropout procedure.
    """

    # Specify the number of units/features incoming.
    dim_in = inputs.get_shape().as_list()[1]

    with tf.variable_scope(name):

        # Weight initialization optimized for ReLUs a la He et al. (2015).
        kernel_init= tf.random_normal_initializer(stddev=sqrt(2.0/dim_in))

        # Functional interface for dense layer.
        l = tf.layers.dense(inputs, units, activation=None, 
                            kernel_initializer=kernel_init)

        # Applying batch normalisaton by Ioffe et al. (2015).
        # l = tf.layers.batch_normalization(l, training=training_phase)

        # Applying activation function.
        l = tf.nn.relu(l)

        # Applying dropout by Srivastava et al. (2014).
        # l = tf.nn.dropout(l, pkeep)

    return l


def dense_nn(nb_features, layer_sizes, nb_labels):
    """
    Utility function that takes a specified NN topology (# units & layers)
    and returns a tensorflow computational graph of a fully connected (dense)
    neural network to be used for training and testing.

    Arguments:
    ----------
        nb_features: int.
            Number of features in labeled data.
        layer_sizes: array-like, shape=[, # layers].
            List with number of units per hidden layer, e.g. [32,16,8,4].
        nb_labels: int.
            Number of labels in labeled data.

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
            nn.err_10pc: tf op (tf.float32)
                Percentage of predictions with relative error of more than 10%.
            nn.err_5pc: tf op (tf.float32)
                Percentage of predictions with relative error of more than 5%.
            nn.jac: tf op(tf.float32)
                Gradient/Jacobian of Output(s) wrt to inputs.

    """

    ## INITIALIZATION

    tf.reset_default_graph()

    # Creating a class of named tuples collecting neural network ops.
    NeuralNetwork = namedtuple('nn', 'inputs, labels, pkeep, training_phase, \
                                predictions, loss, err_10pc, err_5pc, jac')

    # Placeholders for labeled pair of training data.
    inputs = tf.placeholder(tf.float32, [None, nb_features], name='inputs')
    labels = tf.placeholder(tf.float32, [None, nb_labels], name='labels')

    # Batchnorm (Ioffe et al. (2015)).
    training_phase = tf.placeholder(tf.bool, name='training_phase')

    # Dropout (Srivastava et al. (2014)): Probability of keeping nodes.
    pkeep = tf.placeholder(tf.float32, name='pkeep')

    ## CONSTRUCTION OF COMPUTATIONAL GRAPH OF FULLY CONNECTED NN

    nb_hidden_layers = len(layer_sizes)
    layers = []

    # Dealing with special case of first hidden layer.
    first_layer = dense_relu_bn_drop(inputs, layer_sizes[0], training_phase, 
                                     pkeep, 'dense_hidden_0')

    layers.append(first_layer)

    # Dealing with hidden layers between first and final prediction layer.
    for i in range(nb_hidden_layers - 1):

        hidden_layer = dense_relu_bn_drop(layers[i], layer_sizes[i+1],
                                          training_phase, pkeep,
                                          'dense_hidden_%s' % str(i+1))

        layers.append(hidden_layer)

    # Dealing with final prediction layer (linear, no act function)
    dim_in = layer_sizes[-1]

    kernel_init= tf.random_normal_initializer(stddev=sqrt(2.0/dim_in))
    
    prediction_layer = tf.layers.dense(layers[-1], nb_labels, 
                                       activation=None, 
                                       kernel_initializer=kernel_init,
                                       name = 'predictions')

    ## ADDING LOSS & ACCURACY/ERROR TO COMPUTATIONAL GRAPH

    # Define the loss function.
    with tf.name_scope('loss'):

        loss = tf.losses.mean_squared_error(labels, prediction_layer)
        
        tf.summary.scalar('loss', loss)

    # Define accuracy = % of predictions with RE < certain threshold.
    with tf.name_scope('accuracy'):

        # Define the relative error as a metric of accuracy for predictions.
        relative_error = tf.abs((prediction_layer-labels)/labels)

        # Relative error less than 2%
        close_prediction_10pc = tf.greater(relative_error, 0.10)
        err_10pc = tf.reduce_mean(tf.cast(close_prediction_10pc, tf.float32))
        tf.summary.scalar('error_10pc', err_10pc)

        # Relative error less than 1%
        close_prediction_5pc = tf.greater(relative_error, 0.05)
        err_5pc = tf.reduce_mean(tf.cast(close_prediction_5pc, tf.float32))
        tf.summary.scalar('error_5pc', err_5pc)
        
    ## ADDING GRADIENT OF OUTPUT WRT TO INPUTS
    
    with tf.name_scope('jacobian'):
        
        jac = tf.gradients(ys=prediction_layer, xs=inputs)

    ## COLLECTION OPS AND INFOS OF NN IN NAMED TUPLE

    nn = NeuralNetwork(inputs = inputs,
                       labels = labels,
                       pkeep  = pkeep,
                       training_phase = training_phase,
                       predictions = prediction_layer,
                       loss = loss,
                       err_10pc = err_10pc,
                       err_5pc = err_5pc,
                       jac = jac)

    return nn

    