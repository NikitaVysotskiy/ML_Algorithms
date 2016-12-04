import numpy as np
from numpy.linalg import inv
from Gradient_descent import gradient_descent
from Regularization import regularized_gradient_descent, regularized_norm_eq


def main():
    m = 30
    x = [[1]*m, list(range(m))]

    x = np.transpose(x)
    y = list(range(m))

    theta1 = train(x, y, 0.005)
    print("Predictions with gradient descent (expected y = x):")
    print("x = 5.5, y: ", predict(5.5, theta1))
    print("x = 13.2, y: ", predict(13.2, theta1))
    print("x = 138, y: ", predict(138, theta1))

    theta2 = normal_equation(x, y)
    print("Predictions with normal equation (expected y = x):")
    print("x = 5.5, y: ", predict(5.5, theta2))
    print("x = 13.2, y: ", predict(13.2, theta2))
    print("x = 138, y: ", predict(138, theta2))

    print("After regularization")
    theta_reg1 = req_train(x, y, 0.005, 5)
    print("Predictions with regularized gradient descent (expected y = x):")
    print("x = 5.5, y: ", predict(5.5, theta_reg1))
    print("x = 13.2, y: ", predict(13.2, theta_reg1))
    print("x = 138, y: ", predict(138, theta_reg1))

    theta_reg2 = regularized_norm_eq(x, y, 5)
    print("Predictions with regularized normal equation (expected y = x):")
    print("x = 5.5, y: ", predict(5.5, theta_reg2))
    print("x = 13.2, y: ", predict(13.2, theta_reg2))
    print("x = 138, y: ", predict(138, theta_reg2))


def normal_equation(x, y):
    temp = np.transpose(x).dot(x)
    temp = inv(temp)
    temp = temp.dot(np.transpose(x))
    temp = temp.dot(y)
    return temp


def train(x, y, learn_rate):
    theta = [1, 1]
    theta = gradient_descent(theta, learn_rate, len(y), x, y)
    return theta


def req_train(x, y, learn_rate, reg_coef):
    theta = [1, 1]
    theta = regularized_gradient_descent(theta, learn_rate, len(y), x, y, reg_coef)
    return theta


def predict(x, theta):
    return theta[0] + theta[1] * x

if __name__ == '__main__':
    main()
