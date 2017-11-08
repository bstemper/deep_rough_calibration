import os
import sys
import tensorflow as tf
import numpy as np
from helpers import import_labeled_csv_data
from neural_network import rank1_ff_nn


def train(train_set, validation_set, nn_layer_sizes, lr, seed, nb_epochs, 
          mini_batch_size, print_train=False):

    # Initialization.
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed+1)

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
    val_feed_dict = { nn.inputs : validation_set.features,
                      nn.labels : validation_set.labels}

    # Run session through the computational graph.
    with tf.Session() as sess:

        # Init.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(hyp_param_settings, graph=sess.graph)
    
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
            if print_train == True:
                loss, acc_3dp, acc_4dp, acc_5dp = sess.run([nn.loss, nn.acc_3dp, nn.acc_4dp, nn.acc_5dp], feed_dict=val_feed_dict)
                print('Epoch: ', epoch, 'loss:', loss, 'acc3dp: ', acc_3dp, 
                      'acc4dp: ', acc_4dp, 'acc5dp: ', acc_5dp)

            # Save checkpoint files for reuse later.
            saver.save(sess, save_path=hyp_param_settings + '/', global_step=epoch)

            # Stop performing training cycles if network is accurate enough.
            if acc_4dp > 0.99:

                break

        # Saving final model.
        save_path = saver.save(sess, hyp_param_settings + '/' + 'final_model')

        print("Model saved in file: %s" % save_path)


    
