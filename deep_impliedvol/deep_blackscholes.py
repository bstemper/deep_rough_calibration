# LEARN BLACKSCHOLES IMPLIED VOLATILITY WITH A NEURAL NETWORK
# ==============================================================================

import numpy as np
import os
import sys
import tensorflow as tf
import logging
from scipy.stats import norm

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print("Python version " + sys.version)
print("Tensorflow version " + tf.__version__)

# BLACK-SCHOLES SPECIFIC STUFF
# ==============================================================================

def pricer(flag, spot_price, strike, time_to_maturity, vol, risk_free_rate):
    """
    Computes the Black-Scholes price of a European option with possibly
    vectorized inputs.

    :param flag: either "c" for calls or "p" for puts
    :param spot_price: spot price of the underlying
    :param strike: strike price
    :param time_to_maturity: time to maturity in years
    :param vol: annualized volatility assumed constant until expiration
    :param risk_free_rate: risk-free interest rate

    :type flag: string
    :type spot_price: float / numpy array
    :type strike: float / numpy array
    :type time_to_maturity: float / numpy array
    :type vol: float / numpy array
    :type risk_free_rate: float

    :return: Black-Scholes price
    :rtype: float / numpy array

    # Example taking vectors as inputs

    >>> spot = np.array([3.9,4.1])
    >>> strike = 4
    >>> time_to_maturity = 1
    >>> vol = np.array([0.1, 0.2])
    >>> rate = 0

    >>> p = BlackScholes.pricer('c', spot, strike, time_to_maturity, vol, rate)
    >>> expected_price = np.array([0.1125, 0.3751])


    >>> abs(expected_price - p) < 0.0001
    array([ True,  True], dtype=bool)
    """

    # Rename variables for convenience.

    S = spot_price
    K = strike
    T = time_to_maturity
    r = risk_free_rate

    # Compute option price.

    d1 = 1/(vol * np.sqrt(T)) * (np.log(S) - np.log(K) + (r+0.5 * vol**2) * T)

    d2 = d1 - vol * np.sqrt(T)

    if flag == 'c':

        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    elif flag == 'p':

        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def make_batch(nb_samples, min_vol, max_vol):
    """
    Makes a batch of labeled data for learning the Black-Scholes implied
    volatility function. Samples volatilities uniformly from within given 
    interval, computes the corresponding Black-Scholes price and returns both.

    :param nb_samples: number of samples required in batch
    :param min_vol: lower bound of volatilities
    :param max_vol: upper bound of volatilities

    :type nb_samples: int
    :type min_vol: float
    :type max_vol: float

    :return: Black-Scholes prices
    :rtype: numpy array (nb_samples,1)
    :return: corresponding volatilities
    :rtype: numpy array (nb_samples,1)
    """
    
    vol_samples = np.random.uniform(min_vol, max_vol, nb_samples)
    price_samples = pricer(flag, spot, strike, maturity, vol_samples, rate)

    vol_samples = vol_samples.reshape(-1,1)
    price_samples = price_samples.reshape(-1,1)
    
    return price_samples, vol_samples


# TENSORFLOW/TENSORBOARD STUFF
# ==============================================================================

def fc_layer(input, dim_in, dim_out, name='fc_layer'):
    """
    Definition of a fully connected layer.
    """
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=0.1), name='W')
        B = tf.Variable(tf.ones([dim_out])/10, name='B')
        nonlinearity = tf.nn.relu(tf.matmul(input, W) + B)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("nonlinearity", nonlinearity)
        return nonlinearity

def deep_impliedvol_model(learning_rate, fc1_nb, fc2_nb, hparam):
    """
    At the moment only learns function R -> R, price -> vol.
    """

    tf.reset_default_graph()

    # Placeholders for labeled pair of training data.
    X = tf.placeholder(tf.float32, [None, 1], name='input')
    Y_ = tf.placeholder(tf.float32, [None, 1], name='labels')

    # Build computational graph.
    fc1 = fc_layer(X, 1, fc1_nb, 'fc1')
    fc2 = fc_layer(fc1, fc1_nb, fc2_nb, 'fc2')
    Y = fc_layer(fc2, fc2_nb, 1, 'pred')

    # Define the loss function.
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.square(Y-Y_))
        tf.summary.scalar('loss', loss)

    # Define training step.
    with tf.name_scope('training'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Define accuracy to be percentage of predictions within 1E-3 of labels.
    with tf.name_scope('accuracy'):
        close_prediction = tf.less_equal(tf.abs(Y-Y_), 1E-3)
        accuracy = tf.reduce_mean(tf.cast(close_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()

    # Run session through the computational graph.
    with tf.Session() as sess:

        # Initialize all variables in the graph.
        sess.run(tf.global_variables_initializer())

        # Create writers for train and test data.
        train_writer = tf.summary.FileWriter(LOGDIR + hparam + "_train", 
                                             graph=sess.graph)

        test_writer = tf.summary.FileWriter(LOGDIR + hparam + "_test",
                                            graph=sess.graph)

        # Create test sample.
        test_X, test_Y = make_batch(test_batch_size, 1E-10, 2)

        # Run training loops.
        for i in range(nb_training_runs):

            train_X, train_Y = make_batch(train_batch_size, 1E-10, 2)

            # Run backpropagation and summary op.
            _, train_sum = sess.run([train_step, summary], 
                                    feed_dict={X: train_X, Y_: train_Y})

            if i % 50 == 0:
                train_writer.add_summary(train_sum, i)
                test_sum = sess.run(summary, feed_dict={X: test_X, Y_: test_Y})
                test_writer.add_summary(test_sum, i)

            

# MAIN SCRIPTS
# =============================================================================

def make_hparam_string(learning_rate, fc1_nb, fc2_nb):
    return "lr_%.0E,%s,%s" % (learning_rate, fc1_nb, fc2_nb)

def main_single():

    learning_rate = 1E-3
    fc1_nb = 55
    fc2_nb = 10

    hparam = make_hparam_string(learning_rate, fc1_nb, fc2_nb)
    print("Hyperparameter string: %s" %hparam)

    deep_impliedvol_model(learning_rate, fc1_nb, fc2_nb, hparam)

    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)


def main_grid():

    for learning_rate in [1E-3, 1E-4, 1E-5]:
        for fc1_nb in [5, 10, 15, 25, 50, 100, 200, 500]:
            for fc2_nb in [5, 10, 15, 25, 50, 100, 200, 500]:

                if fc2_nb < fc1_nb:

                    hparam = make_hparam_string(learning_rate, fc1_nb, fc2_nb)
                    print(hparam)

                    deep_impliedvol_model(learning_rate, fc1_nb, fc2_nb, hparam)

    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

# CALLING SCRIPTS
# =============================================================================    

if __name__ == '__main__':

    # Parameters of Black-Scholes function (except for the volatility).
    flag = 'c'
    spot = 1
    strike = 1
    maturity = 1
    rate = 0

    # Configuration.
    tf.set_random_seed(42)
    LOGDIR = "/tmp/deep_impliedvol/"
    nb_training_runs = 10**4
    test_batch_size = 1000
    train_batch_size = 100

    # calling

    main_grid()

