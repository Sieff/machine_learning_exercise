# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def load_data(name, m=None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    return (x, y)


def classify(x_train, y_train, x_test, y_test):
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, alpha=0.5)
    plt.show()

    c = np.logspace(-9, 5, 10)
    tuned_parameters = [
        {"kernel": ["rbf"], "C": c},
    ]

    clf = GridSearchCV(
        svm.SVC(gamma="scale"), tuned_parameters, scoring="accuracy", n_jobs=12
    )

    clf.fit(x_train, y_train)

    print(clf.best_params_)
    print("Training accuracy:", clf.score(x_train, y_train))
    print("Test accuracy:", clf.score(x_test, y_test))


if __name__ == '__main__':

    print('Dataset O')
    x_train, y_train = load_data('dataset_O_train.npy')
    x_test, y_test = load_data('dataset_O_test.npy')
    classify(x_train, y_train, x_test, y_test)

    print('Dataset U')
    x_train, y_train = load_data('dataset_U_train.npy')
    x_test, y_test = load_data('dataset_U_test.npy')
    classify(x_train, y_train, x_test, y_test)

    print('Dataset V')
    x_train, y_train = load_data('dataset_V_train.npy')
    x_test, y_test = load_data('dataset_V_test.npy')
    classify(x_train, y_train, x_test, y_test)
