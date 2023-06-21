import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_img_kmeans(img, i):
    img_flattened = img.reshape([img.shape[0] * img.shape[1], 3])
    kmeans = KMeans(n_clusters=i, n_init="auto").fit(img_flattened)
    img_clustered = np.array([kmeans.cluster_centers_[i] for i in kmeans.labels_]).reshape(img.shape)

    plt.imshow(img_clustered)
    plt.title(f'k={i}')
    plt.show()


def cluster_img_pca(img, i):
    r, g, b = cv2.split(img)
    pca_r = PCA(n_components=i).fit(r)
    pca_g = PCA(n_components=i).fit(g)
    pca_b = PCA(n_components=i).fit(b)

    transform_r = pca_r.transform(r)
    transform_g = pca_g.transform(g)
    transform_b = pca_b.transform(b)

    inverse_r = pca_r.inverse_transform(transform_r)
    inverse_g = pca_g.inverse_transform(transform_g)
    inverse_b = pca_b.inverse_transform(transform_b)

    merged = cv2.merge([inverse_r, inverse_g, inverse_b])
    plt.imshow(merged)
    plt.title(f'components={i}')
    plt.show()


if __name__ == '__main__':
    im1 = imread('butterfly.jpg') / 255
    plt.imshow(im1)
    plt.show()

    #for i in range(1, 30, 2):
        #10 - kmeans
        #9 - pca erkennbar
        #40 - pca perfekt
    cluster_img_kmeans(im1, 10)
    cluster_img_pca(im1, 9)
    cluster_img_pca(im1, 40)
    
    im2 = imread('flower.jpg') / 255
    plt.imshow(im2)
    plt.show()

    #for i in range(1, 30, 2):
        #7 - kmeans
        #23 - pca erkennbar
        #50 - pca perfekt
    cluster_img_kmeans(im2, 7)
    cluster_img_pca(im2, 23)
    cluster_img_pca(im2, 50)
    
    im3 = imread('nasa.jpg') / 255
    plt.imshow(im3)
    plt.show()

    #for i in range(1, 30, 2):
        #17 - kmeans
        #17 - pca erkennbar
        #40 - pca perfekt
    cluster_img_kmeans(im3, 17)
    cluster_img_pca(im3, 17)
    cluster_img_pca(im3, 40)
