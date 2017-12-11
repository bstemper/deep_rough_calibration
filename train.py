import tensorflow as tf
import numpy as np
from helpers import *
from neural_network import fully_connected_nn


def train(train_tuple, validation_tuple, hyper_params, nb_epochs, random_seed,
          verbose = False, log_df= None, ckpt_dir = None):
    """

    Arguments:
    ----------
        train_tuple:   named tuple.
            train_tuple.features: array-like, shape=[# samples, # features].
                Features of the data set.
            train_tuple.labels: array-like, shape=[# samples, # labels].
                Labels of the data set.
            train_tuple.nb_features:   integer. 
                Number of features.
            train_tuple.nb_labels: integer.
                Number of labels.
            train_tuple.nb_samples: integer.
                Number of samples.
        validation_tuple:   named tuple.
            validation_tuple.features: array-like, shape=[# samples, # features].
                Features of the data set.
            validation_tuple.labels: array-like, shape=[# samples, # labels].
                Labels of the data set.
            validation_tuple.nb_features:   integer. 
                Number of features.
            validation_tuple.nb_labels: integer.
                Number of labels.
            validation_tuple.nb_samples: integer.
                Number of samples.
        hyper_params: list.
            List of hyperparameters of a neural network.
                layer_sizes: array-like, shape=[, # layers]
                    Number of units per layer, e.g. [32,16,8,4].
                learning_rate: float.
                    Learning rate used for backpropagation.
                mini_batch_size: integer.
                    Size of individual mini-batches used for backpropagation.
        nb_epochs: integer.
            Number of epochs to train the network.
        random_seed: integer.
            Random seed for PRNG, allowing reproducibility of results.
        verbose: boolean. default = False.
            If True, prints intermediate results to console.
        log_df: pandas dataframe, shape=[, nb_layers + 8], default = None
            Pandas df that serves as a log file. If none, df is created.
        ckpt_dir = string, default = None.
            Directory for checkpoints. If none, network is initialized and
            trained. If directory given, last checkpoint is loaded and trained
            further.

    Returns:
    --------
        log_df: pandas dataframe.
            Pandas df log file with training and validation metrics across eps.
        validation_results: array-like, shape=[,3]
            Loss and accuracies on validation set at the last epoch.
    """

    ## PREPROCESSING

    # Set the NumPy PRNG such that feeding of batches may be reproduced.
    np.random.seed(random_seed)

    # Make identifier string for hyperparameter sample.
    hyper_param_str = make_hyper_param_str(hyper_params)
    verbose_print(verbose, hyper_param_str)

    # Splitting hyperparameters into separate variables.
    layer_sizes = hyper_params[0]
    lr = hyper_params[1]
    mini_batch_size = hyper_params[2]

    ## BUILDING THE COMPUTATIONAL GRAPH

    # Build computational graph of a fully connected neural network.
    tf.reset_default_graph()
    nn = fully_connected_nn(train_tuple.nb_features, layer_sizes, 
                            train_tuple.nb_labels, random_seed)

    # Add a training op to computational graph.
    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(lr).minimize(nn.loss)

    # Merge all summary ops in one op for convenience.
    summary = tf.summary.merge_all()
   
    ## INITIATE SESSION

    with tf.Session() as sess:

        # Initialize important tensorflow objects.
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(hyper_param_str, graph=sess.graph)

        # If no checkpoint directory exists, initialize NN. Otherwise, load
        # latest checkpoint to continue training the network.
        if ckpt_dir == None:

            init = tf.global_variables_initializer()
            sess.run(init)

        else:

            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        # If no pandas df is given as log file, create one. Otherwise use 
        # existing one.
        if log_df == None:

            log_df = make_log_df(len(layer_sizes))

        # Define validation feed dictionary to be fed into NN for benchmarking.
        val_feed_dict = { nn.inputs : validation_tuple.features,
                          nn.labels : validation_tuple.labels
                        }

        ## RUN BACKPROPAGATION
            
        # Run through epochs where 1 epoch is 1 full run through training set.
        for epoch in range(1, nb_epochs + 1):

            # Compute how many minibatches to run through.
            nb_mini_batches = int(train_tuple.nb_samples/mini_batch_size)

            # Do random shuffling of the indices of training samples.
            shuffled_indices = np.random.permutation(train_tuple.nb_samples)

            # Running through individual minibatches and doing backprop.
            for i in range(nb_mini_batches):

                mini_batch_indices = shuffled_indices[i:i + mini_batch_size]

                train_feed_dict = { 
                    nn.inputs : train_tuple.features[mini_batch_indices, :],
                    nn.labels: train_tuple.labels[mini_batch_indices, :]
                }

                # Run training/backpropagation op.
                sess.run([train_step], feed_dict=train_feed_dict)

            # Writing intermediate summary statistics to disk for tensorboard.           
            validation_summary = sess.run(summary, feed_dict = val_feed_dict)
            writer.add_summary(validation_summary, epoch)

            # Writing intermediate results to pandas log df.
            metrics_ops = [nn.loss, nn.acc_2pc, nn.acc_1pc]
            train_results = sess.run(metrics_ops, feed_dict=train_feed_dict)
            validation_results = sess.run(metrics_ops, feed_dict=val_feed_dict)

            log_data = layer_sizes + [lr, epoch] + train_results \
                       + validation_results
            log_df.loc[log_df.shape[0]] = log_data
            
            # If verbose is True, print intermediate results.
            epoch_res = 'Ep %i: loss|acc2pc|acc1pc %s,  %s' \
                        % (epoch, train_results, validation_results)

            verbose_print(verbose, epoch_res)

            # Checking conditions for breaking the training
            cond1 = nn_is_fully_trained(log_df, 0.99)
            cond2 = nn_does_not_learn(log_df)

            if (cond1 or cond2) == True:

                break

        # Saving final model.
        save_path = saver.save(sess, hyper_param_str + '/final_model')

        return log_df, validation_results

