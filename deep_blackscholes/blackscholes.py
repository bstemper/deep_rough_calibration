#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# BLACK-SCHOLES FRAMEWORK

# ------------------------------------------------------------------------------
# IMPORTS

import numpy as np
from scipy.stats import norm

# ------------------------------------------------------------------------------
# DEFINITIONS


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
