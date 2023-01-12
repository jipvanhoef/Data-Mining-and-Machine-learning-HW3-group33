#%%
import sklearn as sk 
import numpy as np
import matplotlib.pyplot as plt # data visualization library
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import TruncatedSVD 
faces = fetch_olivetti_faces()
print(faces.DESCR)

D = faces.data



def PCA(X , num_components):
     
    X_meaned = X - np.mean(X , axis = 0)
    truncatedSVD = TruncatedSVD(num_components)
    X_truncated = truncatedSVD.fit_transform(X)

    return (X_meaned.T @ X_truncated).T
    

D_reduced = PCA(D,100)

# Here are the first ten guys of the dataset
fig = plt.figure(figsize=(10, 3))
# for i in range(5):
#     ax = plt.subplot2grid((3, 10), (int(i/10), i-int(i/10)*10))
    
#     ax.imshow(D_reduced[i,:].reshape(64,64), cmap=plt.cm.gray)
#     ax.axis('off')

def showPicture(i):

    ax = plt.subplot2grid((3, 10), (int(i/10), i-int(i/10)*10))
    ax.imshow(D_reduced[i,:].reshape(64,64), cmap=plt.cm.gray)
    ax.axis('off')

showPicture(19)
# %%
