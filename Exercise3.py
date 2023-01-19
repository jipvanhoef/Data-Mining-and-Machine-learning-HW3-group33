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



#get the minimal index
def arg_min_init(indexes, D):
    #init the index and minimal values to the first sum
    min_index = 0
    min_found = np.linalg.norm(D[0, :] - D[0, :]) ** 2

    n = D.shape[0]
    #loop through matrix and get index of minimal sum
    for i in indexes:
        for j in range(n):
            sum = np.linalg.norm(D[j, :] - D[i, :]) ** 2
            if sum < min_found:
                min_index = i
                min_found = sum

    return min_index

#get probabilities
def probabilities(D, X):
    n = D.shape[0]
    probabilities = np.zeros(n)
    denominator = 0
    for j in range(n):
        denominator += dist(D[j, :].T, X)

    for i in range(n):
        numerator = dist(D[i, :].T, X)
        probabilities[i] = numerator / denominator

    return probabilities


#func to get arg min in loop
def loop_arg_min(indexes, D, X):
    min_index = -1
    min_found = 9999999
    n = D.shape[0]
    for i in indexes:
        sum = 0
        for j in range(n):
            mat = add_to_X(X, D[i, :].T)
            sum += dist(D[j, :].T, mat)
        if sum < min_found:
            min_found = sum
            min_index = i
    return min_index

def dist(v, X):
    #check if shape is correct
    assert np.shape(v)[0] == np.shape(X)[0], f"Shape of v: {np.shape(v)} is not equal to shape of X: {np.shape(X)}"
    current_min = 999999999
    #if shape is 1d, just return the norm
    if len(X.shape) == 1:
        return np.linalg.norm(v - X)
    #else loop trough all norms and return smallest
    for i in range(X.shape[1]):
        column_vector = X[:, i]
        d = np.linalg.norm(v - column_vector)
        current_min = min(d, current_min)

    # print("Min found:", min_found)
    return current_min

#func to add array as column to X
def add_to_X(X, column):
    if len(X.shape) == 1:
        mat = np.zeros((X.shape[0], 2))
        mat[:, 0] = X
        mat[:, 1] = column
        return mat
    else:
        return np.append(X, array_to_column(column), axis=1)

#func to cast array to column
def array_to_column(array):
    length = np.shape(array)[0]
    mat = np.zeros((length, 1))
    for i in range(length):
        mat[i, :] = array[i]
    return mat

#init of greedy centroids
def centroids_init(D, r, l=10):
    #get shape of d
    n, d = D.shape
    # random generator to sample candidates
    rng = np.random.default_rng(
        seed=7)
    indexes = rng.integers(low=0, high=n, size=l)
    i = arg_min_init(indexes, D)
    
    X = np.array(D[i, :]).T
    z = 2

    while z <= r:
        probabilities = probabilities(D, X)
        indexes = rng.choice(n, l, p=probabilities)
        i = loop_arg_min(indexes, D, X)
        # print("x: \n", X)

        X = add_to_X(X, D[i, :].T)
        z += 1

    return X

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

X_init = centroids_init(D,r)
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

for neighbour in knn:
    dataID, D, labels, r = generateMoons(0.05,n)
    N = kneighbors_graph(D, neighbour)
    W = 0.5 * (N + N.T)
    Y = spectral_clustering(W, r, centroids_init)
    plt.scatter(D[:, 0], D[:, 1], c=np.argmax(Y, axis=1), s=10)
    plt.title(('%s' % dataID) + ", nrof neigbours: "
              + str(neighbour))
    plt.show()

    nmi = sklearn.metrics.normalized_mutual_info_score(labels, Y[:, 0])
    nmis.append(nmi)

print(nmis)