# Generate labeled Heston data for training and testing.
# ==============================================================================

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_lets_be_rational.exceptions import BelowIntrinsicException

r = robjects.r
r.source("heston.R")
r_pricer = r('HestonCallClosedForm')

def heston_pricer(lambd, vbar, eta, rho, v0, r, tau, S0, K):
    """
    Computes European Call price under Heston dynamics with closedform solution.

    :param lambd: mean-reversion speed
    :param vbar: long-term average volatility
    :param eta: volatility of vol process
    :param rho: correlation between stock and vol
    :param v0: intial volatility
    :param r: risk-free interest rate
    :param tau: time to maturity
    :param S0: initial share price
    :param K: strike price

    :return: Black-Scholes implied volatility
    :rtype: float

    """

    # Compute BS Implied volatility induced by Heston price.
    price = r_pricer(lambd, vbar, eta, rho, v0, r, tau, S0, K)[0]

    try:

        iv = implied_volatility(price, S0, K, tau, r, 'c')

    except BelowIntrinsicException:

        iv = None

        print('Below Intrinsic Value Error. Price %s' % price)

    return iv


def write_to_csv(file_name, data, header):

        np.savetxt(file_name, data, delimiter=',', newline='\n', header=header)

def main():

    # Initialise data array to be exported to CSV.
    data = np.zeros((nb_samples, 8))

    # Calculate Heston prices and associated implied volatilities.
    count = 0

    while count < nb_samples:

        # Sample uniformly from Heston parameter space.
        lambd = np.random.uniform(lambd_bounds[0], lambd_bounds[1])
        vbar = np.random.uniform(vbar_bounds[0], vbar_bounds[1])
        eta = np.random.uniform(eta_bounds[0], eta_bounds[1])
        rho = np.random.uniform(rho_bounds[0], rho_bounds[1])
        v0 = np.random.uniform(v0_bounds[0], v0_bounds[1])

        # Sample uniformly from option parameter space.
        tau = np.random.uniform(tau_bounds[0], tau_bounds[1])
        K = np.random.uniform(K_bounds[0], K_bounds[1])

        # Check for Feller's condition. 
        if 2 * lambd * vbar > eta**2:

            # Calculate Black-Scholes implied vol from Heston price.
            iv = heston_pricer(lambd, vbar, eta, rho, v0, r, tau, S0, K)

            if iv is not None:

                data[count, 0] = lambd
                data[count, 1] = vbar
                data[count, 2] = eta
                data[count, 3] = rho
                data[count, 4] = v0
                data[count, 5] = tau
                data[count, 6] = K
                data[count, 7] = iv

                print('Count %i/%i' % (count + 1, nb_samples))

                # Increase running counter.
                count+=1

    # Write labeled data to .csv file. 
    write_to_csv(csv_file_name, data, 'lambda, vbar, eta, rho, v0, tau, K, iv')

if __name__ == '__main__':

    # Declare seed for sampling from parameter regions.
    np.random.seed(0)

    # Standardised parameters
    S0 = 1
    r = 0

    # Heston parameter bounds by Moodley (2005)
    lambd_bounds = [0, 10]
    vbar_bounds = [0, 1]
    eta_bounds = [0, 5]
    rho_bounds = [-1, 0]
    v0_bounds = [0, 1]

    # Option parameter bounds (based on traded option data)
    tau_bounds = [1/250, 1]
    K_bounds = [0.75, 1.5]

    # Other parameters
    csv_file_name = 'data_uniform.csv'
    nb_samples = 10**3

    main()

