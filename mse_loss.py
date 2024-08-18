"""
This file contains our cost function, which is the mean squared error math equation.

Handles MSE calculations as well as calculating the derivative of the cost function.
"""

import numpy as np


def mse(y_true, y_pred):
    """
    true value vs predicted value
    :param y_true:
    :param y_pred:
    :return:
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
