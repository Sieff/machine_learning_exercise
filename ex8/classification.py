# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def load_data(name, m=None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    return (x, y)


def plot_numbers(numb, tag, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=66)
    ones = [ numb[tag == i][0] for i in range(5) ]
    twos = [ numb[tag == (i+5)][0] for i in range(5) ]
    (fig, axs) = plt.subplots(nrows=2, ncols=5)
    for (ax,i) in zip(axs[0], range(5)):
        ax.imshow(ones[i].reshape(8,8), cmap='gray', vmin=0, vmax=16)
    for (ax,i) in zip(axs[1], range(5)):
        ax.imshow(twos[i].reshape(8,8), cmap='gray', vmin=0, vmax=16)
    plt.show()


def plot_data(x, y):
    plt.scatter([i[0] for i in x], y)
    plt.show()


def plot_points(x, y):
    fig,ax = plt.subplots()
    px = [ i[0] for i in x ]
    py = [ i[1] for i in x ]
    ax.scatter(px,py, c=y)
    fig.show()


def knn_regression(x_train, y_train, x_test, y_test):
    for k in range(1, 11):
        print("k = " + str(k))
        neigh = KNeighborsRegressor(n_neighbors=k)
        neigh.fit(x_train, y_train)
        print("Training score:", neigh.score(x_train, y_train))
        print("Test score:", neigh.score(x_test, y_test))
        plot_data(x_test, neigh.predict(x_test))


def knn_classifier(x_train, y_train, x_test, y_test, title=''):
    print(title)
    k_range = range(1, 20)
    train_scores = []
    test_scores = []
    for k in k_range:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        train_scores.append(neigh.score(x_train, y_train))
        test_scores.append(neigh.score(x_test, y_test))
        #plot_points(x_test, neigh.predict(x_test))

    print('Best k:', np.argmax(test_scores)+1, 'Test Score:', np.max(test_scores))
    plt.plot(k_range, train_scores, color='blue', label='Training score')
    plt.plot(k_range, test_scores, color='red', label='Test score')
    plt.title(title)
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.show()


def lda_classifier(x_train, y_train, x_test, y_test, title=''):
    print(title)

    neigh = LinearDiscriminantAnalysis()
    neigh.fit(x_train, y_train)
    print('Training Score:', neigh.score(x_train, y_train))
    print('Test Score:', neigh.score(x_test, y_test))
    plot_points(x_test, neigh.predict(x_test))


def qda_classifier(x_train, y_train, x_test, y_test, title=''):
    print(title)

    neigh = QuadraticDiscriminantAnalysis()
    neigh.fit(x_train, y_train)
    print('Training Score:', neigh.score(x_train, y_train))
    print('Test Score:', neigh.score(x_test, y_test))
    plot_points(x_test, neigh.predict(x_test))

    
if __name__ == '__main__':
    # Task 1
    print('Dataset R')
    x_train, y_train = load_data('dataset_R_train.npy')
    x_test, y_test = load_data('dataset_R_test.npy')

    plot_data(x_train, y_train)

    knn_regression(x_train, y_train, x_test, y_test)

    # Task 2 + 3 + 4
    print('Dataset E')
    x_train, y_train = load_data('dataset_E_train.npy')
    x_test, y_test = load_data('dataset_E_test.npy')

    plot_points(x_train, y_train)
    plot_points(x_test, y_test)

    knn_classifier(x_train, y_train, x_test, y_test, title='Dataset E - KNN')
    lda_classifier(x_train, y_train, x_test, y_test, title='Dataset E - LDA')
    qda_classifier(x_train, y_train, x_test, y_test, title='Dataset E - QDA')
    
    print('Dataset G')
    x_train, y_train = load_data('dataset_G_train.npy')
    x_test, y_test = load_data('dataset_G_test.npy')

    plot_points(x_train, y_train)
    plot_points(x_test, y_test)

    knn_classifier(x_train, y_train, x_test, y_test, title='Dataset G - KNN')
    lda_classifier(x_train, y_train, x_test, y_test, title='Dataset G - LDA')
    qda_classifier(x_train, y_train, x_test, y_test, title='Dataset G - QDA')

    print('Dataset O')
    x_train, y_train = load_data('dataset_O_train.npy')
    x_test, y_test = load_data('dataset_O_test.npy')

    plot_points(x_train, y_train)
    plot_points(x_test, y_test)

    knn_classifier(x_train, y_train, x_test, y_test, title='Dataset O - KNN')
    lda_classifier(x_train, y_train, x_test, y_test, title='Dataset O - LDA')
    qda_classifier(x_train, y_train, x_test, y_test, title='Dataset O - QDA')

    # Task 5
    print('Dataset digits')
    X, y = load_digits(return_X_y=True)
    plot_numbers(X, y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=66)

    knn_classifier(x_train, y_train, x_test, y_test, title='Dataset digits - KNN')
    lda_classifier(x_train, y_train, x_test, y_test, title='Dataset digits - LDA')
    qda_classifier(x_train, y_train, x_test, y_test, title='Dataset digits - QDA')
