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
train_filename = 'deep_impliedvol/labeled_data_toy/train_uniform.csv'
validation_filename = 'deep_impliedvol/labeled_data_toy/validation_uniform.csv'
test_filename = 'deep_impliedvol/labeled_data_toy/test_uniform.csv'
feature_cols = [0]
label_cols = [1]
random_seed = 42

def main_single():

	nb_epochs = 1000
	mini_batch_size = 100
	nn_layer_sizes = [64, 32]
	lr = 0.00005
	log_name = 'log_test.txt'
	
	# Read training and validation data named tuples into memory.
	train_set = import_labeled_csv_data(train_filename, feature_cols, 
										label_cols)

	validation_set = import_labeled_csv_data(validation_filename, feature_cols, 
											 label_cols)
	
	train(train_set, validation_set, nn_layer_sizes, lr, random_seed, nb_epochs, 
	  mini_batch_size, log_name, print_train=True)

def main_random_search():

	nb_epochs = 10
	mini_batch_size = 100
	max_exp = 10
	nb_learning_rates = 10
	log_name = 'log.txt'

	# Read training and validation data named tuples into memory.
	train_set = import_labeled_csv_data(train_filename, feature_cols, 
										label_cols)

	validation_set = import_labeled_csv_data(validation_filename, feature_cols, 
											 label_cols)

	nn_combinations = [[2**i, 2**j] for i in range(1, max_exp +1) for j in range(1, i+1)]

	for nn_layer_sizes in nn_combinations:

		for lr in 10**np.random.uniform(-6, 1, nb_learning_rates):

			train(train_set, validation_set, nn_layer_sizes, lr, random_seed, nb_epochs, 
	  			  mini_batch_size, log_name, print_train=True)


if __name__ == '__main__':
	
	main_single()

