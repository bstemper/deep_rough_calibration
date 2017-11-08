import os
import sys
import tensorflow as tf
import numpy as np
from helpers import import_labeled_csv_data
from neural_network import rank1_ff_nn

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)


def train(train_filename, validation_filename, feature_cols, label_cols, 
          nn_layer_sizes, lr, seed, nb_epochs):

    # Initialization.
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed+1)

    # Read training and validation data named tuples into memory.
    train_set = import_labeled_csv_data(train_filename, feature_cols, label_cols)
    val_set = import_labeled_csv_data(validation_filename, feature_cols, label_cols) 

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
            loss, acc_3dp, acc_4dp, acc_5dp = sess.run([nn.loss, nn.acc_3dp, nn.acc_4dp, nn.acc_5dp], feed_dict=val_feed_dict)
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

if __name__ == '__main__':

    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    # Configuration.
    ID = 1
    train_filename = 'deep_impliedvol/lab_data_3_1/train_uniform.csv'
    validation_filename = 'deep_impliedvol/lab_data_3_1/validation_uniform.csv'
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

    train(train_filename, validation_filename, feature_cols, label_cols, 
          nn_layer_sizes, lr, seed, nb_epochs)