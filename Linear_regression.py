import numpy as np
from numpy.linalg import inv
from Gradient_descent import gradient_descent


def main():
    m = 30
    x = [[1]*m, list(range(m))]

    x = np.transpose(x)
    y = list(range(m))

    theta = train(x, y, 0.005)
    theta1 = normal_equation(x, y)
    print(predict(5.5, theta))
    print(predict(5.5, theta1))
    print(predict(500, theta))
    print(predict(500, theta1))


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


def predict(x, theta):
    return theta[0] + theta[1] * x

if __name__ == '__main__':
    main()

