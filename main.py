import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd
from helpers import *
from train import train

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.stdout = open('test', 'w', 1)

print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)

# Data config
train_filename = 'labeled_data/heston/training_data.csv'
validation_filename = 'labeled_data/heston/validation_data.csv'
feature_cols = [ _ for _ in range(7)]
label_cols = [7]
random_seed = 46

# Model and training config
nb_epochs = 10
mini_batch_size = 128
layer_sizes = [64, 64, 32]
lr = 1E-5
pkeep = 1

hyper_params = [layer_sizes, lr, mini_batch_size, pkeep]
hyper_param_str = make_hyper_param_str(hyper_params)
ckpt_dir = hyper_param_str

# Read training and validation data named tuples into memory.
train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
validation_tuple = load_labeled_csv(validation_filename, feature_cols, label_cols)

# Run backpropagation training. 
log_df, best_error = train(train_tuple, validation_tuple, hyper_params, 
                           nb_epochs, random_seed, verbose=True)

log_df.to_csv(hyper_param_str + '/log_file.csv')

