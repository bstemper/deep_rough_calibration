# ------------------------ Main script: Bayes Calibration  ------------------ #

import os
import numpy as np
import logging
import tensorflow as tf
import importlib.util
import sys
import pandas as pd
import emcee
import scipy.stats as stats
from math import sqrt

from ann.neural_network import dense_nn
from ann.predict import predict
from ann.helpers import load_labeled_csv

# Important directories
deep_cal_dir = os.path.dirname(os.getcwd())

# Logging stuff
logger = logging.getLogger("benchmark_calibration")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(deep_cal_dir + "/code/logs/benchmark_calibration.log")    
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Training data configuration
train_filename = deep_cal_dir + '/data/rough_bergomi/training_data.csv'
feature_cols = [ _ for _ in range(6)]
label_cols = [6]

# Extract mean and std from training data
train_tuple = load_labeled_csv(train_filename, feature_cols, label_cols)
train_mean = np.mean(train_tuple.features, axis=0)
train_std = np.std(train_tuple.features, axis=0)

# Utility function to standardise inputs for NN training.
def standardise_inputs(test_inputs, train_mean, train_std):
    
    logger.info("Normalizing labeled inputs for feeding in NN.")
    
    test_inputs -= train_mean
    test_inputs /= train_std
    
    return test_inputs

# Utility function to destandardise calibrated inputs
def destandardise_inputs(test_inputs, train_mean, train_std):
    
    test_inputs *= train_std
    test_inputs += train_mean
    
    return test_inputs

# ------------------------- Firing up the network  -------------------------- #

# NN params
layer_sizes = [4096]*4
nb_features = 6
nb_labels = 1

# File directory to saved TF checkpoint files
ckpt_dir = deep_cal_dir + '/data/rough_bergomi/nn/'

# Construction of neural network computational graph
logger.info("Building computational graph of a fully connected neural network.")
nn = dense_nn(nb_features, layer_sizes, nb_labels)

# Restoring weights and biases to NN
logger.info('Starting interactive tensorflow session.')
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
logger.info('Checkpoint loaded. Neural network ready for prediction.')

# ---------------------- Loading calibration data  -------------------------- #

# Reference model parameters
xi = 0.01
H = 0.07
eta = 1.9
rho = -0.9

ref_param = [xi, H, eta, rho]

df = pd.read_csv(deep_cal_dir + 
	'/data/rough_bergomi/jim_rBergomi_bayes_data.csv', index_col=0)

# --------------- Bayesian calibration definitions -------------------------- #

def log_prior_rB(mu):
    
    transf_mu = mu.copy()
    transf_mu[0] = sqrt(transf_mu[0])
    
    bounds = { #bounds are transformed to unit normal bounds (shifted and scaled)
        'sigma_0': [-2.5, 7, 0.3, 0.1],
        'H': [-1.2, 8.6, 0.07, 0.05],
        'eta': [-3, 3, 2.5, 0.5],
        'rho': [-0.25, 2.25, -0.95, 0.2]
    }
    
    return np.sum([stats.truncnorm.logpdf(transf_mu[i], bounds[key][0], bounds[key][1], 
                    bounds[key][2], bounds[key][3]) for i, key in enumerate(bounds)])

def log_prior_heston(mu):
    
    # Heston
    bounds = { 
    'lambda': [0, 10],
    'vbar': [0, 1],
    'eta': [0, 5],
    'rho': [-1, 0],
    'v0': [0,1]
    }

    return np.sum([stats.uniform.logpdf(mu[i], loc=bounds[key][0], scale=bounds[key][1]-bounds[key][0] ) 
                   for i, key in enumerate(bounds)])

def compute_mean(mu, x):
    
    logger.debug('Computing predicted value for inputs.')
    
    maturity, strike = x
    
    model_param = np.tile([mu], (len(df), 1))
    features = np.column_stack((model_param, maturity, strike))  
    features = standardise_inputs(features, train_mean, train_std)
    
    logger.debug('Features normalised. Now predict.')
    mean, _ = predict(features, nn, sess)
    
    logger.debug('Neural network computation successful.')
    
    return mean.reshape(-1)
    
def log_likelihood(mu, x, y, weight, sigma):
  
    mean = compute_mean(mu, x)

    logpdf = stats.multivariate_normal.logpdf(y, mean=mean,
                                              cov = np.sqrt(weight) * sigma) 

    return logpdf

def neg_log_likelihood(mu, x, y, weight, sigma):
    return -log_likelihood(mu, x, y, weight, sigma)

def log_posterior(mu, x, y, weight, sigma):
    
    lp = log_prior_rB(mu)
    
    if not np.isfinite(lp):
        logger.info('Mu: {}. Logpos: -inf'.format(mu))
        return - np.inf
    
    result = lp + log_likelihood(mu, x, y, weight, sigma)

    logger.info('Mu: {}. Logpos: {}'.format(mu, result))
    
    return result

# ----------------------- Bayesian calibration run -------------------------- #


n_dim = 4       # number of parameters
n_walkers = 40   # number of MCMC walkers
n_burn = 250    # "burn-in" period to let chains stabilize
n_steps = 300   # number of MCMC steps to take after burn-in

np.random.seed(0)

pos = [ref_param + 1e-3*np.random.randn(n_dim) for i in range(n_walkers)]



sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, 
                                args=((df.maturity.values, df.strike.values), 
                                	   df.mid.values, df.weight.values, 
                                       df.sigma.values))



sampler.run_mcmc(pos, n_burn + n_steps)

print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))

np.save('chain_market', sampler.chain)





