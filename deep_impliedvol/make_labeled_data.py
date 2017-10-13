# Generate labeled Black-Scholes data for training and testing.
# ==============================================================================

import numpy as np
from scipy.stats import norm


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
    Makes a single batch of labeled data for learning the Black-Scholes implied
    volatility function. Samples volatilities uniformly from within given 
    interval, computes the corresponding Black-Scholes price and returns the
    labeled pair as a numpy array.

    :param nb_samples: number of samples required in batch
    :param min_vol: lower bound of volatilities
    :param max_vol: upper bound of volatilities

    :type nb_samples: int
    :type min_vol: float
    :type max_vol: float

    :return: labeled pair (price, vol)
    :rtype: numpy array (nb_samples,2)
    """
    
    vol_samples = np.random.uniform(min_vol, max_vol, nb_samples)
    price_samples = pricer(flag, spot, strike, maturity, vol_samples, rate)

    labeled_pair = np.stack((price_samples, vol_samples), axis=1)
    
    return labeled_pair

def write_to_csv(file_name, data, header):

        np.savetxt(file_name, data, delimiter=',', newline='\n', header=header)

def main():

        # Sample labeled data.
        data = make_batch(nb_samples, min_vol, max_vol)

        # Write labeled data to .csv file. 
        write_to_csv(csv_file_name, data, 'price, vol')

if __name__ == '__main__':

    np.random.seed(0)

    # Parameters of Black-Scholes function (except for the volatility).
    flag = 'c'
    spot = 1
    strike = 1
    maturity = 1
    rate = 0
    min_vol = 1E-5
    max_vol = 2

    # Other parameters
    csv_file_name = 'data_uniform.csv'
    nb_samples = 10**6
  

    main()

