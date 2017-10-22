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


def make_batch(S, r, flag, K_bounds, tau_bounds, vol_bounds, csv_file_name, nb_samples):

    # Sample uniformly from input space.
    vol_samples = np.random.uniform(vol_bounds[0], vol_bounds[1], nb_samples)
    strike_samples = np.random.uniform(K_bounds[0], K_bounds[1], nb_samples)
    tau_samples = np.random.uniform(tau_bounds[0], tau_bounds[1], nb_samples)

    # Compute Black-Scholes price. 
    price_samples = pricer(flag, S, strike_samples, tau_samples, vol_samples, r)

    # Construct labeled pairs.
    S = np.tile(int(S), nb_samples)
    r = np.tile(int(r), nb_samples)
    labeled_pair = np.stack((S, r, strike_samples, tau_samples, price_samples, 
                            vol_samples), axis=1)
    
    return labeled_pair

def write_to_csv(file_name, data, header):

        np.savetxt(file_name, data, delimiter=',', newline='\n', header=header)

def main():

        # Sample labeled data.
        data = make_batch(S, r, flag, K_bounds, tau_bounds, vol_bounds, csv_file_name, nb_samples)

        # Write labeled data to .csv file.
        header = 'spot, rate, strike, time to maturity, price, vol'
        write_to_csv(csv_file_name + ".csv", data, header)

if __name__ == '__main__':

    # Declare seed for sampling from parameter regions.
    np.random.seed(1337)

    # Fixed parameters.
    S = 1           # spot
    r = 0           # rate
    flag = 'c'

    # Varying parameters.
    K_bounds = [0.75, 1.25]     # strike
    tau_bounds = [0, 90/365]    # time to maturity 
    vol_bounds = [1E-4, 2]      # volatility

    # Other parameters
    csv_file_name = 'data_uniform'
    nb_samples = 10**6
  
    main()

