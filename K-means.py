from random import randrange
import numpy as np


def main():
    m = 10
    x = [[randrange(0, 10) for i in range(m)] + [randrange(11, 21) for i in range(m)],
         [randrange(0, 10) for i in range(m)] + [randrange(11, 21) for i in range(m)],
         [randrange(0, 2) for i in range(m * 2)]]
    print("X before clustering:\n", np.transpose(x))
    num_of_clusters = 2
    x = clusterise(x, num_of_clusters)

    print("X, splitted on {0} clusters (3rd num of each point refers to num of cluster):\n".format(num_of_clusters), x)


def count_distance(x, k):
    return np.sqrt(np.sum((np.array(k) - x) ** 2, axis=1))


def clusterise(x, num_of_clusters):
    m = len(x[0])
    x = np.transpose(x)
    k = []

    for i in range(num_of_clusters):
        k.append(x[randrange(m)][:-1])

    for n in range(100):
        for point in x:
            distances = count_distance(point[:-1], k)
            min_dist = np.argmin(distances)
            point[2] = min_dist

        clusters = []

        for i in range(len(k)):
            clusters.append([])

        for point in x:
            clusters[point[2]].append(point[:-1])

        for i in range(len(k)):
            for j in range(len(k[i])):
                k[i][j] = np.average(np.transpose(clusters[i])[j]) if len(clusters[i]) != 0 else 0

    return x

if __name__ == '__main__':
    main()