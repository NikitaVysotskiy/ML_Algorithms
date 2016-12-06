from random import randrange
import math
import numpy as np


def main():

    m = 10
    training_set = np.transpose([[randrange(0, 10) for i in range(2 * m)],
                                [randrange(0, 6) for i in range(m)] + [randrange(6, 11) for i in range(m)],
                                [0] * m + [1] * m])

    params = count_params(training_set)
    # print(params)
    test = [[0, 0], [0, 7], [10, 3], [10, 10]]
    predictions = get_all_predictions(params, test)
    for i in range(len(test)):
        print("Point {0} might be: {1}".format(test[i], "negative" if predictions[i] == 0 else "positive"))
    # print(predictions)


def separate(train_set):
    separated = {}
    for i in range(len(train_set)):
        temp = train_set[i]
        if temp[-1] not in separated:
            separated[temp[-1]] = []
        separated[temp[-1]].append(temp)
    return separated


def count_params(train_set):
    separated = separate(train_set)
    params = {}
    for class_value, instances in separated.items():
        params[class_value] = [(mean(x_i), standard_dev(x_i)) for x_i in zip(*instances)]
        del params[class_value][-1]
    return params


def mean(nums):
    return sum(nums) / float(len(nums))


def standard_dev(nums):
    avg = mean(nums)
    variance = sum([pow(x - avg, 2) for x in nums]) / float(len(nums) - 1)
    return math.sqrt(variance)


def prob(x, mean, stdev):
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exp


def count_class_probs(params, inp):
    probs = {}
    for class_value, class_params in params.items():
        probs[class_value] = 1
        for i in range(len(class_params)):
            mean, stdev = class_params[i]
            x = inp[i]
            probs[class_value] *= prob(x, mean, stdev)
    return probs


def predict(summaries, inp):
    probs = count_class_probs(summaries, inp)
    best_label, best_prob = None, -1
    for class_value, prob in probs.items():
        if best_label is None or prob > best_prob:
            best_prob = prob
            best_label = class_value
    return best_label


def get_all_predictions(params, test_set):
    predictions = []
    for i in range(len(test_set)):
        res = predict(params, test_set[i])
        predictions.append(res)
    return predictions


if __name__ == '__main__':
    main()


