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
    X_truncated = truncatedSVD.fit_transform(X_meaned)
    #X_truncated[19] = [-1 for i in range(50)]

    return truncatedSVD.inverse_transform(X_truncated)
    

def showPicture(i):
    ax = plt.subplot2grid((3, 10), (int(i/10), i-int(i/10)*10))
    ax.imshow(D_reduced[i,:].reshape(64,64), cmap=plt.cm.gray)
    ax.axis('off')
#A
D_reduced = PCA(D,5)
fig = plt.figure(figsize=(50, 15))
for i in range(5):
    showPicture(i)
#B
X_meaned = D - np.mean(D , axis = 0)
truncatedSVD = TruncatedSVD(5)
X_truncated = truncatedSVD.fit_transform(X_meaned)
print(X_truncated[0])

#C
D_reduced = PCA(D,25)
showPicture(19)
D_reduced = PCA(D,50)
showPicture(19)
D_reduced = PCA(D,100)

#D
X_meaned = D - np.mean(D , axis = 0)
truncatedSVD = TruncatedSVD(50)
X_truncated = truncatedSVD.fit_transform(X_meaned)
X_truncated[19] = [-1 for i in range(50)]
D_reduced = truncatedSVD.inverse_transform(X_truncated)
showPicture(19)
# %%
