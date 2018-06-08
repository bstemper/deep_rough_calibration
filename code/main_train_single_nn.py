# ---------------- Main script: Single training run ----------------------- #

import numpy as np
import tensorflow as tf
import pandas as pd
import sys
from ann.helpers import *
from ann.train import *
from hyperdash import Experiment
import os
import logging
import logging.config
from os.path import dirname as up

deep_cal_dir = up(os.getcwd())

# --------------------- Logging stuff ------------------------------------- #

# Logging stuff
logger = logging.getLogger('deep_cal')
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/single_training_nn.log")    
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


# --------------------- Configuration ------------------------------------- #

# Labeled data configuration
train_filename = deep_cal_dir + '/data/rough_bergomi/training_data.csv'
validation_filename = deep_cal_dir + '/data/rough_bergomi/validation_data.csv'
feature_cols = [ _ for _ in range(6)]
label_cols = [6]
# Model and training configuration
random_seed = 10
nb_epochs = 300
mini_batch_size = 2048
layer_sizes = [64]*10
lr = 1E-3
pkeep = 1

hyper_params = [layer_sizes, lr, mini_batch_size, pkeep]
hyper_param_str = make_hyper_param_str(hyper_params)
ckpt_dir = hyper_param_str

# --------------------- Preprocessing ------------------------------------- #
 
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

# ------------------------ Training --------------------------------------- #

hd_exp = Experiment(hyper_param_str)

# Run backpropagation training. 
df, best_error = train(train_tuple, validation_tuple, hyper_params, nb_epochs, 
                       random_seed, hd_exp, deep_cal_dir + '/code/')

logger.info('Writing log dataframe to csv on disk.')
df.to_csv(hyper_param_str + '/log_file.csv')

# Finish Hyperdash experiment.
hd_exp.end()

logger.info("PROGRAM END.")

