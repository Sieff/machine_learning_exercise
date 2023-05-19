# -*- coding: utf-8 -*-
from itertools import cycle

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def load_data(name, m=None, rng=None):
    data = np.load(name)
    x = data[:, :-1]
    y = data[:, -1]

    if not m is None:
        if rng is None: rng = np.default_rng(seed=66)
        idx = rng.choice(m, size=len(x), replace=False)
        x = x[idx]
        y = y[idx]

    return (x, y)


def plot(x, y, w=None, sigma=None):
    '''
    only for plotting 2D data
    '''

    plt.plot(x, y, '.r', markersize=8, label='Samples')

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X_plot = np.vander(x_plot, deg)

        # set plotting range properly
        plt.ylim((np.min(y) * 1.2, np.max(y) * 1.2))

        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

        # also plot confidence intervall
        if not sigma is None:
            plt.plot(x_plot, np.dot(X_plot, w) + sigma, linewidth=2, color='tab:cyan')
            plt.plot(x_plot, np.dot(X_plot, w) - sigma, linewidth=2, color='tab:cyan')

    plt.tight_layout()
    plt.savefig('fig.pdf')

    plt.show()


if __name__ == '__main__':
    x, y = load_data('dataset_sparse_train.npy')
    x_test, y_test = load_data('dataset_sparse_test.npy')
    clf = linear_model.LassoCV(eps=1e-3, n_alphas=1000)
    clf.fit(x, y)
    print(clf.coef_)
    alphas, coefs, _ = clf.path(x,y)
    print('Training score', clf.score(x, y))
    print('Test score', clf.score(x_test, y_test))

    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_lasso = -np.log10(alphas)
    for coef_l, c in zip(coefs, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)

    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso Path")
    plt.axis("tight")
    plt.show()

