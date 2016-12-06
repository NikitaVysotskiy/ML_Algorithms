from random import randrange
from math import sqrt
import numpy as np

def main():
    m = 10
    x = [[randrange(0, 10) for i in range(m)] + [randrange(20, 30) for i in range(m)],
        [randrange(0, 10) for i in range(m)] + [randrange(20, 30) for i in range(m)]]

    print(x)

    clusterise(x)

def clusterise(x):
    # k = 2

    k = [[x[0][randrange(len(x[0]))], x[1][randrange(len(x[1]))]],
         [x[0][randrange(len(x[0]))], x[1][randrange(len(x[1]))]]]
    print(k)

    cluster0 = []
    cluster1 = []


    for i in range(len(x[0])):
        dist_to_0 = sqrt((x[0][i] - k[0][0]) ** 2 + (x[1][i] - k[0][1]) ** 2)
        dist_to_1 = sqrt((x[0][i] - k[1][0]) ** 2 + (x[1][i] - k[1][1]) ** 2)
        if dist_to_0 > dist_to_1:
            cluster0.append([x[0][i], x[1][i]])
        else:
            cluster1.append([x[0][i], x[1][i]])

    cluster0 = np.transpose(cluster0)
    cluster1 = np.transpose(cluster1)
    k[0][0] = np.average(cluster0[0])
    k[0][1] = np.average(cluster0[1])
    k[1][0] = np.average(cluster1[0])
    k[1][1] = np.average(cluster1[1])

    for n in range(1000):
        for i in range(len(cluster0[0])):
            dist_to_0 = sqrt((cluster0[0][i] - k[0][0]) ** 2 + (cluster0[1][i] - k[0][1]) ** 2)
            dist_to_1 = sqrt((cluster0[0][i] - k[1][0]) ** 2 + (cluster0[1][i] - k[1][1]) ** 2)
            if dist_to_0 > dist_to_1:
                list(cluster0).append([cluster0[0][i], cluster0[1][i]])
            else:
                list(cluster1).append([cluster0[0][i], cluster0[1][i]])

        for i in range(len(cluster0[0])):
            dist_to_0 = sqrt((cluster1[0][i] - k[0][0]) ** 2 + (cluster1[1][i] - k[0][1]) ** 2)
            dist_to_1 = sqrt((cluster1[0][i] - k[1][0]) ** 2 + (cluster1[1][i] - k[1][1]) ** 2)
            if dist_to_0 > dist_to_1:
                list(cluster0).append([cluster1[0][i], cluster1[1][i]])
            else:
                list(cluster1).append([cluster1[0][i], cluster1[1][i]])

        k[0][0] = np.average(cluster0[0])
        k[0][1] = np.average(cluster0[1])
        k[1][0] = np.average(cluster1[0])
        k[1][1] = np.average(cluster1[1])

    print(cluster0)
    print(cluster1)

    print(k)

if __name__ == '__main__':
    main()