import os
import sys
import tensorflow as tf
import numpy as np
from helpers import load_labeled_csv
from neural_network import rank1_ff_nn


def train(train_tuple, validation_tuple, nn_layer_sizes, lr, random_seed, 
          nb_epochs, mini_batch_size, log_df, print_log=False,
          ckpt_dir = None):

    # Initialization.
    tf.reset_default_graph()
    np.random.seed(random_seed)

    # Build the computational graph of a feed-forward NN.
    nn = rank1_ff_nn(train_tuple.nb_features, nn_layer_sizes, 
                     train_tuple.nb_labels, random_seed)

    # Build the training op.
    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(lr).minimize(nn.loss)

    # Stringify network configuration.
    net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
    hyp_param_settings = net_config + ",lr_%.5E" % (lr) 

    # Collect all summary ops in one op.
    summary = tf.summary.merge_all()

    # Build the validation set dictionary.
    val_feed_dict = { nn.inputs : validation_tuple.features,
                      nn.labels : validation_tuple.labels}

    # Run session through the computational graph.
    with tf.Session() as sess:

        # Initialization
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(hyp_param_settings, graph=sess.graph)

        if ckpt_dir is not None:

            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        else:

            init = tf.global_variables_initializer()
            sess.run(init)
        
        # Perform training cycles.
        for epoch in range(nb_epochs):

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

                # Run training step (which includes backpropagation).
                sess.run([train_step], feed_dict=train_feed_dict)

            # Writing results on validation set to disk.
            validation_summary = sess.run(summary, feed_dict = val_feed_dict)
            writer.add_summary(validation_summary, epoch)

            # Writing intermediate results to pandas log df.
            train_results = sess.run([nn.loss, nn.acc_2pc, 
                                      nn.acc_1pc], feed_dict=train_feed_dict)
            val_results = sess.run([nn.loss, nn.acc_2pc, 
                                    nn.acc_1pc], feed_dict=val_feed_dict)

            log_data = nn_layer_sizes + [lr] + train_results + val_results
            
            log_df.loc[log_df.shape[0]] = log_data
            
            if print_log == True:
                print('Epoch: %i, train loss/acc2pc/acc1pc: %s, validation loss/acc2pc/acc1pc: %s'
                       % (epoch, train_results, val_results))

            # Stop performing training cycles if network performs well on validation set.
            if val_results[2] > 0.99:

                break

        # Saving final model.
        save_path = saver.save(sess, hyp_param_settings + '/' + 'final_model')

        return log_df
    
