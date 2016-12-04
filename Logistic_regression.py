from random import randrange
import numpy as np
from Gradient_descent import gradient_descent


def main():
    m = 100
    x = [[1] * (2 * m),
         [randrange(0, 6) for i in range(m)] + [randrange(6, 10) for i in range(m)],
         [randrange(0, 6) for i in range(m)] + [randrange(6, 10) for i in range(m)]]
    x = np.transpose(x)
    y = [0] * m + [1] * m
    learn_rate = 0.005
    theta1 = train(x, y, learn_rate)
    xt1 = [1, 3]
    xt2 = [7, 8]
    print(theta1)
    print(predict(xt1, theta1))
    print(predict(xt2, theta1))


def train(x, y, learn_rate):
    theta = [1, 1, 1]
    theta = gradient_descent(theta, learn_rate, len(y), x, y, "log")
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

