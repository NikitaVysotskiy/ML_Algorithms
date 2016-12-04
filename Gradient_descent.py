import numpy as np
from math import e


def gradient_descent(theta, rate, m, x, y, alg="lin"):
    num_of_iterations = 10000
    x_t = np.transpose(x)
    for i in range(num_of_iterations):
        if alg == "log":
            h_x = log_hypothesis(x, theta)
        else:
            h_x = linear_hypothesis(x, theta)

        gradient = np.dot(x_t, h_x - y) / m
        theta -= rate * gradient
    return theta


def linear_hypothesis(x, theta):
    return np.dot(x, theta)


def log_hypothesis(x, theta):
    return 1 / (1 + e ** (- np.dot(x, theta)))
