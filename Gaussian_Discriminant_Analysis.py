from random import randrange

import numpy as np


X = np.array([[randrange(1000) for i in range(100)],
              [randrange(100) for i in range(100)],
              [randrange(10) for i in range(100)],
              [randrange(10) for i in range(100)]])

y = [randrange(1, 4) for i in range(100)]

mean_vectors = []

for cl in range(1, 4):
    for i in range(0, 4):
        mean_vectors.append(np.mean([X[i][j] for j in range(len(X[i])) if y[j] == cl]))


mean_vectors = np.reshape(mean_vectors, (3, 4))

#print(X.T)
#print(y)
#print("Mean vectors:\n", mean_vectors)

S_W = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
S_I = np.zeros((4, 4))
X = X.T

#np.set_printoptions(precision=4)

for i in range(len(X)):
        x_t = np.reshape(X[i], (4, 1))
        m_t = np.reshape(mean_vectors[y[i] - 1], (4, 1))
        S_W += (x_t - m_t).dot((x_t - m_t).T)


#print("S_W", S_W)

S_B = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

m = np.reshape(np.mean(X, axis=0), (4, 1))
n = [0,0,0]

for i in range(len(X)):
    n[y[i]-1] += 1


for i in range(len(mean_vectors)):

        m_i = np.reshape(mean_vectors[i], (4, 1))
        S_B += n[i] * (m_i - m).dot((m_i - m).T)


#print(np.mean(X, axis=0))
#print("S_B", S_B)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(4,1)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in decreasing order:')
for i in eig_pairs:
    print(i[0])


eigv_sum = sum(eig_vals)

W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', W.real)


X_lda = X.dot(W)

print("Transformed dataset:\n", X_lda)



