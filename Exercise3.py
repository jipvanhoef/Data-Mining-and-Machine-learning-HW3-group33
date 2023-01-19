from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.autograd import Variable
import torchvision
import os
import random
import pandas as pd # pandas is a data manipulation library
import numpy as np #provides numerical arrays and functions to manipulate the arrays efficiently
import matplotlib.pyplot as plt # data visualization library
import sklearn
from sklearn.datasets import fetch_olivetti_faces

import scipy

def generateMoons(epsilon, n):
    moons, labels = sklearn.datasets.make_moons(n_samples=n, noise=epsilon, random_state=7)
    return "moons", moons, labels, 2
def generateBlobs(epsilon, n):
    blobs, labels = sklearn.datasets.make_blobs(n_samples=n,centers=3, cluster_std=[epsilon + 1, epsilon + 2.5, epsilon + 0.5])
    return "blobs", blobs, labels, 3

def init_centroids_greedy_pp(D, r, l=10):
    """
        :param r: (int) number of centroids (clusters)
        :param D: (np-array) the data matrix
        :param l: (int) number of centroid candidates in each step
        :return: (np-array) 'X' the selected centroids from the dataset
    """
    rng = np.random.default_rng(
        seed=7)
    # use this random generator to sample the candidates (sampling according to given probabilities
    # can be done via rng.choice(..))
    n, d = D.shape

    indexes = rng.integers(low=0, high=n, size=l)

    # print(indexes, D)
    i = arg_min_init(indexes, D)

    # print("Found min index:", i)
    # print(indexes)

    X = np.array(D[i, :]).T

    s = 2

    while s <= r:
        # print("step s:", s)
        probabilities = get_probabilities(D, X)
        indexes = rng.choice(n, l, p=probabilities)
        i = arg_min_loop(indexes, D, X)
        # print("x: \n", X)

        X = append_to_X(X, D[i, :].T)
        s += 1

    return X

def get_probabilities(D, X):
    # print("get probabilities")
    n = D.shape[0]
    probabilities = np.zeros(n)
    denominator = 0
    for j in range(n):
        denominator += dist(D[j, :].T, X)

    for i in range(n):
        numerator = dist(D[i, :].T, X)
        probabilities[i] = numerator / denominator

    return probabilities

# Line 3 of the pseudocode
def arg_min_init(indexes, D):
    min_index = -1
    min_found = 9999999
    n = D.shape[0]
    for i in indexes:
        for j in range(n):
            sum = np.linalg.norm(D[j, :] - D[i, :]) ** 2
            if sum < min_found:
                min_index = i
                min_found = sum

    return min_index

def arg_min_loop(indexes, D, X):
    min_index = -1
    min_found = 9999999
    n = D.shape[0]
    for i in indexes:
        sum = 0
        for j in range(n):
            mat = append_to_X(X, D[i, :].T)
            sum += dist(D[j, :].T, mat)
        if sum < min_found:
            min_found = sum
            min_index = i
    return min_index

def dist(v, X):
    assert np.shape(v)[0] == np.shape(X)[0], f"Shape of v: {np.shape(v)} is not equal to shape of X: {np.shape(X)}"
    min_found = 999999999
    # print(X.shape)

    if len(X.shape) == 1:
        return np.linalg.norm(v - X)

    for i in range(X.shape[1]):
        column_vector = X[:, i]
        d = np.linalg.norm(v - column_vector)
        min_found = min(d, min_found)

    # print("Min found:", min_found)
    return min_found

def append_to_X(X, column):
    if len(X.shape) == 1:
        mat = np.zeros((X.shape[0], 2))
        mat[:, 0] = X
        mat[:, 1] = column
        return mat
    else:
        return np.append(X, cast_array_to_column_vector(column), axis=1)

def cast_array_to_column_vector(arra):
    length = np.shape(arra)[0]
    mat = np.zeros((length, 1))
    for i in range(length):
        mat[i, :] = arra[i]
    return mat

def spectral_clustering(W,r, X_init):
    '''
        :param W: (np-array) nxn similarity/weighted adjacency matrix
        :param r: (int) number of centroids (clusters)
        :param X_init: (function) the centroid initialization function 
        :return: (np-array) 'Y' the computed cluster assignment matrix
    '''
    L = np.diag(np.array(W.sum(0))[0]) - W
    # print("L: ", L)
    Lambda, V = scipy.sparse.linalg.eigsh(L, k=r+1, which="SM")
    print("V: ", V)
    A = V[:,1:]
    initial_points = X_init(A,r)
    X, Y = kmeans(A, r, initial_points)
    return Y

def RSS(D,X,Y):
    return np.sum((D- Y@X.T)**2)

def getY(labels):
    '''
        Compute the cluster assignment matrix Y from the categorically encoded labels
    '''
    Y = np.eye(max(labels)+1)[labels]
    return Y
def update_centroid(D,Y):
    cluster_sizes = np.diag(Y.T@Y).copy()
    cluster_sizes[cluster_sizes==0]=1
    return D.T@Y/cluster_sizes
def update_assignment(D,X):
    dist = np.sum((np.expand_dims(D,2) - X)**2,1)
    labels = np.argmin(dist,1)
    return getY(labels)
def kmeans(D,r, X_init, epsilon=0.00001, t_max=10000):
    X = X_init.copy()
    Y = update_assignment(D,X)
    rss_old = RSS(D,X,Y) +2*epsilon
    t=0
    #Looping as long as difference of objective function values is larger than epsilon
    while rss_old - RSS(D,X,Y) > epsilon and t < t_max-1:
        rss_old = RSS(D,X,Y)
        X = update_centroid(D,Y)
        Y = update_assignment(D,X)
        t+=1
    print(t,"iterations")
    return X,Y

n=500
dataID, D, labels, r = generateBlobs(0.05,n)

X_init = init_centroids_greedy_pp(D,r)
X,Y = kmeans(D,r, X_init)

fig = plt.figure()
ax = plt.axes()
ax.axis('equal')
ax.scatter(D[:, 0], D[:, 1], c=np.argmax(Y,axis=1), s=10)
ax.scatter(X_init.T[:, 0], X_init.T[:, 1], c='red', s=50, marker = 'D')
ax.scatter(X.T[:, 0], X.T[:, 1], c='blue', s=50, marker = 'D')

dataID, D, labels, r = generateMoons(0.05,n)

from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph, NearestNeighbors

# Implement here the computation of W as knn graph
knn = [15, 25, 30, 35]

nmis = []
n=500

from sklearn.neighbors import KNeighborsTransformer
# dataID, D, labels, r = generateMoons(0.05,n)

# neigh = NearestNeighbors(radius=1.6)
# neigh.fit(D)
# rng = neigh.radius_neighbors(D)
# print(rng)

for neighbours in knn:
    dataID, D, labels, r = generateMoons(0.05,n)
    N = kneighbors_graph(D, neighbours)
    W = 0.5 * (N + N.T)
    # print(W.shape)
    Y = spectral_clustering(W, r, init_centroids_greedy_pp)
    plt.scatter(D[:, 0], D[:, 1], c=np.argmax(Y, axis=1), s=10)
    plt.title(('%s' % dataID) + ", nrof neigbours: "
              + str(neighbours))
    plt.show()
    # print(Y)

    nmi = sklearn.metrics.normalized_mutual_info_score(labels, Y[:, 0])
    # print("nmi score of moons with knn:", nmi)
    nmis.append(nmi)

print(nmis)