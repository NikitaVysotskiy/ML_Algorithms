import numpy as np
from random import randrange


def main():

    x = np.array([[randrange(6) for i in range(10)],
                  [randrange(6) for i in range(10)],
                  [randrange(6) for i in range(10)]])

    mean = [[], [], []]
    for i in range(len(mean)):
        mean[i] = np.average(x[i])

    cov_matrix = np.cov([x[0, :], x[1, :], x[2, :]])

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

    eigen_pairs.sort()

    w = np.hstack((eigen_pairs[0][1].reshape(3, 1), eigen_pairs[1][1].reshape(3, 1)))

    transformed = np.transpose(w).dot(x)

    print("Reduced data from 3d to 2d:\n", transformed)

if __name__ == '__main__':
    main()