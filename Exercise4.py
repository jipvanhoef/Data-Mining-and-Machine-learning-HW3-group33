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

def matrix_completion(D, r, tmax, labda):
    n, d = D.shape
    X, Y = InitRandom(n, d, r)
    O = IndicatorNonzero(D)
    t = 1
    while t < tmax:
        for k in range(d):
            OXk = np.diag(O[:,k])
            X[:, k] = D[k:,].T @ Y @ np.linalg.inv(Y.T @ OXk @ Y + labda*np.eye(r))
        for i in range(n):
            OYi = np.diag(O[i,:])
            Y[i,:] = D[i]*X @ np.linalg.inv(X.T @ OYi @ X + labda*np.eye(r))
        t += 1
    return X, Y

# Run matrix completion algorithm
X, Y = matrix_completion(D, r, tmax, labda)

# Calculate MSEO at each iteration
O = IndicatorNonzero(D)
mses = []
for t in range(1, tmax+1):
    mse = 1/np.sum(O) * np.linalg.norm(D - O*(Y @ X.T))**2
    mses.append(mse)

# Plot MSEO vs. iteration
plt.plot(range(1, tmax+1), mses)
plt.xlabel('Iteration')
plt.ylabel('MSEO')
plt.show()
