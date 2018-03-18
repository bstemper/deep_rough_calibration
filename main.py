# Main Script Single Run
# # # # # # # # # # # # #

import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import sys
from helpers import *
from train import train

# Logging stuff
logger = logging.getLogger("deep_cal")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logger.log")    
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Labeled data configuration
train_filename = 'labeled_data/heston/training_data.csv'
validation_filename = 'labeled_data/heston/validation_data.csv'
feature_cols = [ _ for _ in range(7)]
label_cols = [7]

# Model and training configuration
random_seed = 46
nb_epochs = 5
mini_batch_size = 128
layer_sizes = [64, 64, 32]
lr = 1E-5
pkeep = 1

hyper_params = [layer_sizes, lr, mini_batch_size, pkeep]
hyper_param_str = make_hyper_param_str(hyper_params)
ckpt_dir = hyper_param_str

# # # # # # # # # # # # # # 
 
logger.info("PROGRAM START with " + hyper_param_str)
logger.debug("Python version" + sys.version)
logger.debug("Tensorflow version" + tf.__version__)

# Read training and validation data named tuples into memory.
logger.info("Importing and normalizing input labeled data.")
train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

# Normalize training and validation data by training statistics
train_mean = np.mean(train_tuple.features, axis=0)
train_std = np.std(train_tuple.features, axis=0)

train_tuple.features -= train_mean
train_tuple.features /= train_std

validation_tuple.features -= train_mean
validation_tuple.features /= train_std

logger.info('Finished importing and normalization of input data.')

# Run backpropagation training. 
df, best_error = train(train_tuple, validation_tuple, hyper_params, nb_epochs, 
                       random_seed, verbose=True)

logger.info('Writing log dataframe to csv on disk.')
df.to_csv(hyper_param_str + '/log_file.csv')

logger.info("Done!")

