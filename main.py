import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from helpers import load_labeled_csv, create_log_df
from neural_network import rank1_ff_nn
from train import train

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)


# Configuration.
train_filename = 'deep_impliedvol/labeled_data/train_uniform.csv'
validation_filename = 'deep_impliedvol/labeled_data/validation_uniform.csv'
test_filename = 'deep_impliedvol/labeled_data/test_uniform.csv'
feature_cols = [0, 1, 2]
label_cols = [3]
random_seed = 42

def main_single():

    nb_epochs = 100
    mini_batch_size = 100
    nn_layer_sizes = [32, 32, 16]
    lr = 5.34770E-05
    log_name = 'run1'
    
    # Read training and validation data named tuples into memory.
    train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
    validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

    # Stringify network configuration.
    net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
    hyp_param_settings = net_config + ",lr_%.5E" % (lr) 

    # Create log dataframe object.
    log_df = create_log_df(log_name, len(nn_layer_sizes))

    # Run backpropagation training. 
    log_df = train(train_tuple, validation_tuple, nn_layer_sizes, lr, 
                   random_seed, nb_epochs, mini_batch_size, log_df, 
                   ckpt_dir = hyp_param_settings, print_log=True)

    log_df.to_csv(hyp_param_settings + '/' + log_name + '.csv')

def main_multiple():

    nb_epochs = 10
    mini_batch_size = 100
    log_name = 'run0'
    nb_hidden = 3

    nn_combinations = [[32, 32, 16]]

    learning_rates = 10**np.random.uniform(-8, -4, 10)

    # Read training and validation data named tuples into memory.
    train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
    validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

    # Run backpropagation training through all combinations.
    for nn_layer_sizes in nn_combinations:

        for lr in learning_rates:

            # Create log dataframe object.
            log_df = create_log_df(log_name, nb_hidden)

            # Stringify network configuration.
            net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
            hyp_param_settings = net_config + ",lr_%.5E" % (lr) 

            log_df = train(train_tuple, validation_tuple, nn_layer_sizes, lr, 
                           random_seed, nb_epochs, mini_batch_size, log_df, 
                           ckpt_dir=None, print_log=True)

            log_df.to_csv(hyp_param_settings + '/' + log_name + '.csv')


def main_random_search():

    nb_epochs = 10
    mini_batch_size = 100
    max_exp = 9
    nb_learning_rates = 10
    nb_hidden = 3
    log_name = 'run0'

    # Read training and validation data named tuples into memory.
    train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
    validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

    nn_combinations = [[2**i, 2**j, 2**k] for i in range(2, max_exp +1) 
                        for j in range(2, i+1) for k in range(2, j+1)]

    learning_exps = np.random.uniform(-8, -4, nb_learning_rates)

    # Run backpropagation training through all combinations.
    for nn_layer_sizes in nn_combinations:

        for lr in 10**learning_exps:

            # Create log dataframe object.
            log_df = create_log_df(log_name, nb_hidden)

            # Stringify network configuration.
            net_config = str(nn_layer_sizes)[1:-1].replace(" ", "")
            hyp_param_settings = net_config + ",lr_%.5E" % (lr) 

            log_df = train(train_tuple, validation_tuple, nn_layer_sizes, lr, 
                           random_seed, nb_epochs, mini_batch_size, log_df, 
                           print_log=True)

            log_df.to_csv(hyp_param_settings + '/' + log_name + '.csv')

if __name__ == '__main__':
    
    main_single()

