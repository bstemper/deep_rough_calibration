import os
import sys
import tensorflow as tf
import numpy as np
from helpers import import_labeled_csv_data
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

	nb_epochs = 100000
	mini_batch_size = 2
	nn_layer_sizes = [32, 16]
	lr = 0.005
	log_name = 'log_test.txt'
	
	# Read training and validation data named tuples into memory.
	train_set = import_labeled_csv_data(train_filename, feature_cols, 
										label_cols)

	validation_set = import_labeled_csv_data(validation_filename, feature_cols, 
											 label_cols)
	
	train(train_set, validation_set, nn_layer_sizes, lr, random_seed, nb_epochs, 
	  mini_batch_size, log_name, print_log=True)

def main_random_search():

	nb_epochs = 10
	mini_batch_size = 100
	max_exp = 9
	nb_learning_rates = 10
	log_name = 'hyp_sweep_log.txt'

	# Read training and validation data named tuples into memory.
	train_set = import_labeled_csv_data(train_filename, feature_cols, 
										label_cols)

	validation_set = import_labeled_csv_data(validation_filename, feature_cols, 
											 label_cols)

	nn_combinations = [[2**i, 2**j, 2**k] for i in range(2, max_exp +1) 
						for j in range(2, i+1) for k in range(2, j+1)]

	for nn_layer_sizes in nn_combinations:

		for lr in 10**np.random.uniform(-6, 1, nb_learning_rates):

			train(train_set, validation_set, nn_layer_sizes, lr, random_seed, nb_epochs, 
	  			  mini_batch_size, log_name, print_log=True)


if __name__ == '__main__':
	
	main_random_search()

