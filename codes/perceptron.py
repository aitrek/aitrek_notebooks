import numpy as np

import matplotlib.pyplot as plt
from aitrek.classifier import Perceptron

data = np.loadtxt("../dataset/linearly_separable_X_200x2_Y_+1_-1.csv",
                  delimiter=",")
X = data[:, :2]
Y = data[:, -1]

# Fit model using primal algorithm.
perceptron = Perceptron(eta=0.01, dual=False)
perceptron.fit(X, Y)

w = perceptron.w
b = perceptron.b

fig1 = plt.figure(figsize=(8, 6))
choice = Y == 1
plt.scatter(X[:, 0][choice], X[:, 1][choice], marker="+")
choice = Y == -1
plt.scatter(X[:, 0][choice], X[:, 1][choice], marker=".")

X0L = np.linspace(start=min(X[:, 0]), stop=max(X[:, 0]), num=100)
X1L = -(X0L * w[0] + b) / w[1]

plt.plot(X0L, X1L, color="r")
plt.show()


# Fit model using dual form algorithm.
perceptron = Perceptron(eta=0.01, dual=True)
perceptron.fit(X, Y)

w = perceptron.w
b = perceptron.b

fig2 = plt.figure(figsize=(8, 6))
choice = Y == 1
plt.scatter(X[:, 0][choice], X[:, 1][choice], marker="+")
choice = Y == -1
plt.scatter(X[:, 0][choice], X[:, 1][choice], marker=".")

X0L = np.linspace(start=min(X[:, 0]), stop=max(X[:, 0]), num=100)
X1L = -(X0L * w[0] + b) / w[1]

plt.plot(X0L, X1L, color="r")
plt.show()

