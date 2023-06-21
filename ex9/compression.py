import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


if __name__ == '__main__':
    im1 = imread('butterfly.jpg') / 255
    plt.imshow(im1)
    plt.show()
    
    im2 = imread('flower.jpg') / 255
    plt.imshow(im2)
    plt.show()
    
    im3 = imread('nasa.jpg') / 255
    plt.imshow(im3)
    plt.show()
