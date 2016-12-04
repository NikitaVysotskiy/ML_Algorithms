import numpy as np
from math import e
from numpy.linalg import inv
from Gradient_descent import linear_hypothesis, log_hypothesis


def regularized_gradient_descent(theta, rate, m, x, y, reg_coef, alg="lin"):
    num_of_iterations = 10000
    x_t = np.transpose(x)
    for i in range(num_of_iterations):
        if alg == "log":
            h_x = log_hypothesis(x, theta)
        else:
            h_x = linear_hypothesis(x, theta)

        gradient = np.dot(x_t, h_x - y) / m
        theta = np.array(theta).astype(float)
        theta = theta * (1 - (rate * float(reg_coef)) / float(m)) - rate * gradient
    return theta


def regularized_norm_eq(x, y, reg_coef):
        temp = np.transpose(x).dot(x)
        l = np.zeros((len(temp), len(temp)))
        np.fill_diagonal(l, 1)
        l[0][0] = 0
        l = l.astype(int)
        temp += np.dot(reg_coef, l)
        temp = inv(temp)
        temp = temp.dot(np.transpose(x))
        temp = temp.dot(y)
        return temp
