# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def load_data(name, m=None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    return (x, y)


def plot_numbers(numb, tag, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed=66)
    ones = numb[tag == 0]
    fives = numb[tag == 1]

    (fig, axs) = plt.subplots(nrows=2, ncols=4)

    indx = rng.integers(len(ones), size=4)
    for (ax,i) in zip(axs[0], indx):
        ax.imshow(ones[i].reshape(28,28), cmap='gray', vmin=0, vmax=1)

    indx = rng.integers(len(fives), size=4)
    for (ax,i) in zip(axs[1], indx):
        ax.imshow(fives[i].reshape(28,28), cmap='gray', vmin=0, vmax=1)
    
    plt.show()


def regression(model):
    x_train, y_train = load_data('dataset_numbers_train.npy')
    plot_numbers(x_train, y_train)

    clf = model
    clf.fit(x_train, y_train)

    train_predictions = clf.predict(x_train)
    correct = train_predictions == y_train
    score = np.sum(correct) / len(y_train)

    print('Train score (accuracy) is:', score)
    print('Number of incorrectly labled datapoints:', len(y_train) - np.sum(correct))

    x_test, y_test = load_data('dataset_numbers_test.npy')

    test_predictions = clf.predict(x_test)
    correct = test_predictions == y_test
    score = np.sum(correct) / len(y_test)

    print('Test score (accuracy) is:', score)
    print('Number of incorrectly labled datapoints:', len(y_test) - np.sum(correct))


if __name__ == '__main__':
    print('Logistical regression:')
    regression(LogisticRegression(random_state=0))

    print('SVM:')
    regression(svm.SVC())
