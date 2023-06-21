import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def distance(x1, x2):
    return np.linalg.norm(np.subtract(x1, x2))


def distances(x, centers):
    return np.array([distance(x, c) for c in centers])


def clusters(X, centers):
    clusters = [[] for _ in centers]
    for x in X:
        clusters[np.argmin(distances(x, centers))].append(x)
    return clusters


def newCenters(C):
    centers = []
    for xs in C:
        centers.append(np.mean(xs, axis=0))
    return centers


def print_clusters(C):
    for i, c in enumerate(C):
        print(f'Cluster {i}:', [list(xs) for xs in c])


def print_centers(c):
    print(f'Centers:', [list(xs) for xs in c])


def ex1():
    X = np.array([[1,2], [2,1], [4,3], [5,4], [6,5], [7,6], [9,8], [10,7]])
    centers = np.array([[2, 1], [10, 7]])

    for i in range(2):
        print('Iteration:', i+1)
        cs = clusters(X, centers)
        print_clusters(cs)
        centers = newCenters(cs)
        print_centers(centers)


def ex2():
    X = np.array([[1,2], [2,1], [4,3], [5,4], [6,5], [7,6], [9,8], [10,7]])
    centers = np.array([[1,1], [5,4], [9,8]])

    for i in range(2):
        print('Iteration:', i+1)
        cs = clusters(X, centers)
        print_clusters(cs)
        centers = newCenters(cs)
        print_centers(centers)


if __name__ == '__main__':
    ex1()
    ex2()
