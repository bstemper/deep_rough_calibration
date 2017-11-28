import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from helpers import import_labeled_csv_data, create_log_df
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

	nb_epochs = 3
	mini_batch_size = 100
	nn_layer_sizes = [16, 16, 16]
	lr = 5.99533E-05
	log_name = 'test_log'
	
	# Read training and validation data named tuples into memory.
	train_tuple = import_labeled_csv_data(train_filename, feature_cols, label_cols)
	validation_tuple = import_labeled_csv_data(validation_filename, feature_cols, label_cols)

	# Create log dataframe object.
	log_df = create_log_df(log_name, len(nn_layer_sizes))

	# Run backpropagation training.	
	log_df = train(train_tuple, validation_tuple, nn_layer_sizes, lr, 
					 random_seed, nb_epochs, mini_batch_size, log_df, 
					 print_log=True)

	log_df.to_csv(log_name + '.csv')

def main_random_search():

	nb_epochs = 2
	mini_batch_size = 100
	max_exp = 9
	nb_learning_rates = 10
	nb_hidden = 3
	log_name = 'hyp_sweep_log'

	# Read training and validation data named tuples into memory.
	train_tuple = import_labeled_csv_data(train_filename, feature_cols, label_cols)
	validation_tuple = import_labeled_csv_data(validation_filename, feature_cols, label_cols)

	nn_combinations = [[2**i, 2**j, 2**k] for i in range(2, max_exp +1) 
						for j in range(2, i+1) for k in range(2, j+1)]

	# Create log dataframe object.
	log_df = create_log_df(log_name, nb_hidden)

	# Run backpropagation training through all combinations.
	for nn_layer_sizes in nn_combinations:

		for lr in 10**np.random.uniform(-6, 1, nb_learning_rates):

			log_df = train(train_tuple, validation_tuple, nn_layer_sizes, lr, 
				  		   random_seed, nb_epochs, mini_batch_size, log_df, 
				  		   print_log=True)

			log_df.to_csv(log_name + '.csv')

if __name__ == '__main__':
	
	main_random_search()

