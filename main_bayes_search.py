# Generic imports
import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time

# Scikit-Optimize imports
from skopt import gp_minimize, dump

# Custom imports
from helpers import *
from neural_network import *
from train import *

# Log file name
filename = 'heston_3layer_1'

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.stdout = open(filename + '.txt', 'w', 1)

print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)

## BAYES WRAPPER FOR GP MINIMIZE
## ===========================================================================

def train_bayes(params):
    """
    Wrapper around train function to serve as objective function for Gaussian
    optimization in scikit-optimize routine gp_minimize.

    Arguments:
    ----------
        params: list, shape=[nb_layers + 3,]
        List of search space dimensions. Entries have to be tuples 
        (lower_bound, upper_bound) for Reals or Integers.

        - 

    Returns:
    --------
        valid_loss: float.
            Loss on the validation set at the end of the training.

    """
    # Translate params into format understood by train function
    layer_sizes = (2**np.array(params[:3])).tolist()
    learning_rate = 10**params[3]
    mini_batch_size = 2**params[4]
    pkeep = params[5]
    hyper_params = [layer_sizes, learning_rate, mini_batch_size, pkeep]
    hyper_param_str = make_hyper_param_str(hyper_params)

    # Call train function
    tic = time.time()
    log_df, validation_results = train(train_tuple, validation_tuple, hyper_params, 
                                       nb_epochs, random_seed, verbose=True)
    
    elapsed_time = time.time()-tic

    # Setting the objective function
    valid_loss, valid_err2pc, valid_err1pc = validation_results
    objective = valid_err2pc
    
    # Writing Pandas log file to csv file on disk.
    log_df.to_csv(hyper_param_str + '/log_file.csv')
    
    return objective, elapsed_time

## IMPLIED VOLATILITY
## ===========================================================================

# # Data directories & data extraction params
# train_filename = 'deep_impliedvol/labeled_data/train_uniform.csv'
# validation_filename = 'deep_impliedvol/labeled_data/validation_uniform.csv'
# test_filename = 'deep_impliedvol/labeled_data/test_uniform.csv'
# feature_cols = [0, 1, 2]
# label_cols = [3]

# # Read training and validation data named tuples into memory.
# train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
# validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

# random_seed = 42
# nb_epochs = 50

# # Search space
# space = [(1,14), 
#          (1,14),
#          (1,14),
#          (-7.0, -1),
#          (3,11),
#          (0.25,1.0)
#         ]

## HESTON
## ===========================================================================

# Data directories & data extraction params
train_filename = 'deep_heston/labeled_data/train_uniform.csv'
validation_filename = 'deep_heston/labeled_data/validation_uniform.csv'
test_filename = 'deep_heston/labeled_data/test_uniform.csv'
feature_cols = [i for i in range(7)]
label_cols = [7]

random_seed = 100
nb_epochs = 15

# Search space
space = [(1,10), 
         (1,10),
         (1,10),
         (-7.0, -1.0),
         (3,10),
         (0.25,1.0)
        ]

# Read training and validation data named tuples into memory.
train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

## OPTIMIZATION
## ===========================================================================

res_gp = gp_minimize(train_bayes, space, n_calls=200, n_random_starts=20, acq_func='EIps',
                     n_jobs=-1, verbose=True, random_state=random_seed)

dump(res_gp, filename + '.gz')

