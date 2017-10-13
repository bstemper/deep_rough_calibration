# Generate labeled Heston data for training and testing.
# ==============================================================================

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from py_vollib.black_scholes.implied_volatility import implied_volatility

r = robjects.r
r.source("heston.R")
r_pricer = r('HestonCallClosedForm')

def HestonCallClosedForm(lambd, vbar, eta, rho, v0, r, tau, S0, K):
    """
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
    price = r_pricer(lambd, vbar, eta, rho, v0, r, tau, S0, K)[0]

    iv = implied_volatility(price, S0, K, tau, r, 'c')

    return iv

def write_to_csv(file_name, data, header):

        np.savetxt(file_name, data, delimiter=',', newline='\n', header=header)

def main():

    # Initialise data to be written to CSV
    data = np.zeros((nb_samples, 8))

    # Sample uniformly from multidimensional input space.
    lambd = np.random.uniform(lambd_bounds[0], lambd_bounds[1], nb_samples)
    vbar = np.random.uniform(vbar_bounds[0], vbar_bounds[1], nb_samples)
    eta = np.random.uniform(eta_bounds[0], eta_bounds[1], nb_samples)
    rho = np.random.uniform(rho_bounds[0], rho_bounds[1], nb_samples)
    v0 = np.random.uniform(v0_bounds[0], v0_bounds[1], nb_samples)

    tau = np.random.uniform(tau_bounds[0], tau_bounds[1], nb_samples)
    K = np.random.uniform(K_bounds[0], K_bounds[1], nb_samples)

    for i in range(nb_samples):

        iv = HestonCallClosedForm(lambd[i], vbar[i], eta[i], rho[i], v0[i], r, tau[i], S0, K[i])

        data[i, 0] = lambd[i]
        data[i, 1] = vbar[i]
        data[i, 2] = eta[i]
        data[i, 3] = rho[i]
        data[i, 4] = v0[i]
        data[i, 5] = tau[i]
        data[i, 6] = K[i]
        data[i, 7] = iv

        # Write labeled data to .csv file. 
        write_to_csv(csv_file_name, data, 'lambda, vbar, eta, rho, v0, tau, K, iv')

if __name__ == '__main__':

    # Declare seed for sampling from parameter regions.
    np.random.seed(0)

    # Standardised parameters
    S0 = 1
    r = 0

    # Parameter bounds by Moodley (2005)
    lambd_bounds = [0, 10]
    vbar_bounds = [0, 1]
    eta_bounds = [0, 5]
    rho_bounds = [-1, 0]
    v0_bounds = [0, 1]

    # Option parameter bounds (based on traded option data)
    tau_bounds = [0, 1]
    K_bounds = [0.5, 1.5]

    # Other parameters
    csv_file_name = 'training_uniform.csv'
    nb_samples = 10**3

    main()

