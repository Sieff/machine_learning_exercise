import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np


def reduce_dimension(X):
    pca = PCA(n_components=2).fit(X)

    pca_transform = pca.transform(X)

    plt.scatter(pca_transform[:, 0], pca_transform[:, 1])
    plt.title("PCA")
    plt.show()

    mds_transform = MDS(n_components=2).fit_transform(X)

    plt.scatter(mds_transform[:, 0], mds_transform[:, 1])
    plt.title("MDS")
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X = np.load('./box.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    reduce_dimension(X)

    X = np.load('./spring_1.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    reduce_dimension(X)
    
    X = np.load('./spring_2.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    reduce_dimension(X)
