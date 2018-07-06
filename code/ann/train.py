# ------------------- Neural network training ----------------------------- #

import tensorflow as tf
import numpy as np
import logging
from .helpers import *
from .neural_network import dense_nn

# Logging stuff
logger = logging.getLogger("deep_cal.train")


def train(train_tuple, validation_tuple, hyper_params, nb_epochs, seed,
          hd_exp, project_dir, log_df= None, ckpt_dir = None):
    """

    Arguments:
    ----------
        train_tuple:   named tuple
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
                pkeep: float. Has to lie between 0 and 1.
                    Probabability of keeping neurons in dropout.
        nb_epochs: integer.
            Number of epochs to train the network.
        seed: integer.
            Random seed for PRNG, allowing reproducibility of results.
        hd_exp: Hyperdash experiment object
            Needed for hyperparameter and general training visualization.
        project_dir: string.
            Project directory to write to.
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
        best_error: float
            Best err10pc on validation set among epochs.
    """

    ## PREPROCESSING

    # Set the NumPy PRNG such that feeding of batches may be reproduced.
    np.random.seed(0)

    # Make identifier string for hyperparameter sample.
    hyper_param_str = make_hyper_param_str(hyper_params)

    # Splitting hyperparameters into separate variables.
    layer_sizes = hyper_params[0]
    lr = hyper_params[1]
    mini_batch_size = hyper_params[2]
    pkeep = hyper_params[3]

    ## BUILDING THE COMPUTATIONAL GRAPH

    # Build computational graph of a fully connected neural network.
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    logger.info("Building computational graph of a fully connected neural network.")

    nn = dense_nn(train_tuple.nb_features, layer_sizes, train_tuple.nb_labels)

    logger.info('Done.')

    # Add training op to computational graph and include Batch norm ops.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Ensures that we execute the update_ops before performing the train_step.
    with tf.control_dependencies(update_ops):
        
        train_step = tf.train.AdamOptimizer(lr).minimize(nn.loss)

    # Merge all summary ops in one op for convenience.
    summary = tf.summary.merge_all()
   
    ## INITIATE SESSION

    with tf.Session() as sess:

        # Initialize important tensorflow objects.
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(project_dir + '/' + hyper_param_str, 
                                       graph=sess.graph)

        # If no checkpoint directory exists, initialize NN. Otherwise, load
        # latest checkpoint to continue training the network.
        if ckpt_dir == None:

            logger.info('No checkpoint exists. Initialize NN.')
            init = tf.global_variables_initializer()
            sess.run(init)

        else:

            logging.info('Checkpoint exists. Loading checkpoint.')
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        # If no pandas df is given as log file, create one. Otherwise use 
        # existing one.
        if log_df == None:

            logger.info('Pandas DF for logging does not exist. Creating now.')
            log_df = make_log_df(len(layer_sizes))

        # Define feeds to be fed into NN for benchmarking.
        train_testing_feed = { 
                    nn.inputs : train_tuple.features,
                    nn.labels : train_tuple.labels,
                    nn.training_phase: False,
                    nn.pkeep: 1
                    }

        val_testing_feed = { 
                        nn.inputs : validation_tuple.features,
                        nn.labels : validation_tuple.labels,
                        nn.training_phase: False,
                        nn.pkeep  : 1
                        }

        ## RUN BACKPROPAGATION

        logger.info('Starting backpropagation.')
            
        # Run through epochs where 1 epoch is 1 full run through training set.
        for epoch in range(1, nb_epochs + 1):

            # Compute how many minibatches to run through.
            nb_mini_batches = int(train_tuple.nb_samples/mini_batch_size)

            # Do random shuffling of the indices of training samples.
            shuffled_indices = np.random.permutation(train_tuple.nb_samples)

            # Running through individual minibatches and doing backprop.
            for i in range(nb_mini_batches):

                start_index = i * mini_batch_size
                stop_index = start_index + mini_batch_size

                mini_batch_indices = shuffled_indices[start_index:stop_index]

                logger.debug('Iter {}. \n Input \n {}. \n Label \n {}'.format(
                            i, train_tuple.features[mini_batch_indices, :],
                            train_tuple.labels[mini_batch_indices, :]))

                train_training_feed = { 
                    nn.inputs : train_tuple.features[mini_batch_indices, :],
                    nn.labels : train_tuple.labels[mini_batch_indices, :],
                    nn.training_phase: True,
                    nn.pkeep  : pkeep
                    }

                # Run training/backpropagation op.
                sess.run([train_step], feed_dict=train_training_feed)

            # Writing intermediate summary statistics to disk for tensorboard.           
            validation_summary = sess.run(summary, feed_dict=val_testing_feed)
            writer.add_summary(validation_summary, epoch)

            # Writing intermediate results to pandas log df.
            metrics_ops = [nn.loss, nn.err_10pc, nn.err_5pc]
            train_results = sess.run(metrics_ops, feed_dict=train_testing_feed)
            validation_results = sess.run(metrics_ops, feed_dict=val_testing_feed)

            log_data = layer_sizes + [lr, pkeep, epoch] + train_results \
                       + validation_results
            log_df.loc[log_df.shape[0]] = log_data
            best_error = log_df.val_err5pc.min()
            
            epoch_res = 'Ep %i: loss|err10pc|err5pc %s,  %s. Best err5pc: %.4f' \
                        % (epoch, train_results, validation_results,
                           best_error)

            # Logging results to hyperdash and logging csv.
            hd_exp.metric('train loss', train_results[0])
            hd_exp.metric('train error 10%', train_results[1])
            hd_exp.metric('train error 5%', train_results[2])
            hd_exp.metric('validation loss', validation_results[0])
            hd_exp.metric('validation error 10%', validation_results[1])
            hd_exp.metric('validation error 5%', validation_results[2])
            hd_exp.metric('best 5% val error', best_error)

            logger.info(epoch_res)

            # Checking conditions for breaking the training
            cond1 = nn_is_fully_trained(log_df, 0.01)
            cond2 = nn_does_not_learn(log_df)

            if cond2:

                break

            elif cond1:

                # Saving final model.
                logger.info('Saving trained model to disk.')
                save_path = saver.save(sess, project_dir + '/' + hyper_param_str + 
                                       '/final_model')

                break

            elif log_df.val_err5pc.values[-1] == best_error:

                # Saving final model.
                logger.info('Saving best model to disk.')
                save_path = saver.save(sess, project_dir + '/' + hyper_param_str + 
                                       '/best_model')

        return log_df, best_error

