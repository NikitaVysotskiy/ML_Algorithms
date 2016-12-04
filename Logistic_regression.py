from random import randrange
import numpy as np
from Gradient_descent import gradient_descent
from Regularization import regularized_gradient_descent


def main():
    m = 100
    x = [[1] * (2 * m),
         [randrange(0, 6) for i in range(m)] + [randrange(6, 10) for i in range(m)],
         [randrange(0, 6) for i in range(m)] + [randrange(6, 10) for i in range(m)]]
    x = np.transpose(x)
    y = [0] * m + [1] * m
    theta1 = train(x, y, 0.005)
    xt1 = [1, 3]
    xt2 = [7, 8]
    print("Decision boundary: {0} + {1} * x1 + {2} * x2 > 0".format(theta1[0], theta1[1], theta1[2]))
    print("Point with x1 = {0}, x2 = {1} might be: ".format(xt1[0], xt1[1]), predict(xt1, theta1))
    print("Point with x1 = {0}, x2 = {1} might be: ".format(xt2[0], xt2[1]), predict(xt2, theta1))

    print("\nAfter regularization:")
    theta2 = reg_train(x, y, 0.005, 5)
    print("Decision boundary: {0} + {1} * x1 + {2} * x2 > 0".format(theta2[0], theta2[1], theta2[2]))
    print("Point with x1 = {0}, x2 = {1} might be: ".format(xt1[0], xt1[1]), predict(xt1, theta2))
    print("Point with x1 = {0}, x2 = {1} might be: ".format(xt2[0], xt2[1]), predict(xt2, theta2))


def train(x, y, learn_rate):
    theta = [1, 1, 1]
    theta = gradient_descent(theta, learn_rate, len(y), x, y, "log")
    return theta


def reg_train(x, y, learn_rate, reg_coef):
    theta = [1, 1, 1]
    theta = regularized_gradient_descent(theta, learn_rate, len(y), x, y, reg_coef, "log")
    return theta


def predict(x_test, theta):
    temp = x_test[:]
    temp.insert(0, 1)
    if np.dot(temp, theta) < 0:
        return "negative"
    else:
        return "positive"


if __name__ == '__main__':
    main()
