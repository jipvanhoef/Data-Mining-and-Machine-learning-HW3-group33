import sklearn as sk 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # data visualization library
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import TruncatedSVD 

np.random.seed()

# lets explore movies.csv
movies= pd.read_csv('data/movies.csv')
ratings=pd.read_csv('data/ratings.csv',sep=',')

#init values
tmax = 100
r = 5
labda = 0.0001

#convert sparse representation to data matrix
df_movie_ratings = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)  #fill unobserved entries with μ

#filter only movies with more then 200 ratings
keep_movie = np.sum(df_movie_ratings!=0,0)>200
df_D = df_movie_ratings.loc[:,keep_movie]

#filter out all users with less then 5 movie ratings
keep_user = np.sum(df_D!=0,1)>5
df_D = df_D.loc[keep_user,:]

#convert to data matrix
D = df_D.to_numpy()


def InitRandom(n, d, r ):
    X = np.random.rand(d, r)
    Y = np.random.rand(n, r)
    return X, Y

def IndicatorNonzero(D):
    return ( D != 0).astype(int)

def stoppingcriteria(mses):
    if len(mses) < 10:
        return False
    else:
        delta = []
        for i in range(1,12):
            delta.append((mses[len(mses)-i-1]) - mses[len(mses)-i])
        delta = np.mean(delta)
        if delta < 0:
            return True
        else:
            return False

mses = []

def matrix_completion(D, r, tmax=100, labda=0.00001):
    n, d = D.shape #n = 344, d = 18
    X, Y = InitRandom(n, d, r) # = (d,r), Y = (n,r)
    O = IndicatorNonzero(D)
    t = 1

    while t < tmax:
        for k in range(d):
            OXk = np.diag(O[:, k])
            X[k,] = D[:,k].T.dot(Y) @ np.linalg.inv(Y.T @ OXk @ Y + labda*np.eye(r))
        for i in range(n):
            OYi = np.diag(O[i,:])
            result = D[i,].dot(X) @ np.linalg.inv(X.T @ OYi @ X + labda*np.eye(r))
            Y[i,] =  result
            
        mse = (1/np.linalg.norm(O,ord=1)) * (np.linalg.norm(D - np.multiply(O,(Y @ X.T)), ord = 2))**2

        mses.append(mse)
        t += 1
        # if(stoppingcriteria(mses=mses)):
        #     break

    return X, Y, t

# Run matrix completion algorithm
X, Y, t = matrix_completion(D, r, tmax, labda)

# Plot MSEO vs. iteration
plt.plot(range(1, t), mses)
plt.xlabel('Iteration')
plt.ylabel('MSEO')
plt.show()