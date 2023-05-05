# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def load_data(name, m = None, rng = None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    if not m is None:
        if rng is None: rng = np.default_rng(seed = 66)
        idx = rng.choice(m, size = len(x), replace = False)
        x = x[idx]
        y = y[idx]

    return (x, y)

def plot(x, y, w = None, sigma = None):
    '''
    only for plotting 2D data
    '''
    
    plt.plot(x, y, '.r', markersize = 8, label = 'Samples')

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X_plot = np.vander(x_plot, deg)

        # set plotting range properly
        plt.ylim((np.min(y)*1.2, np.max(y)*1.2))

        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

        # also plot confidence intervall
        if not sigma is None:
            plt.plot(x_plot, np.dot(X_plot, w)+sigma, linewidth=2, color='tab:cyan')
            plt.plot(x_plot, np.dot(X_plot, w)-sigma, linewidth=2, color='tab:cyan')

    plt.tight_layout()
    plt.savefig('fig.pdf')

    plt.show()


def regression(x, y):

    # code your solution here
    n, m = x.shape  # for generality
    X0 = np.ones((n, 1))
    x = np.hstack((x, X0))


    w = np.linalg.lstsq(x, y, rcond=None)
    
    return w[0]


def regressionLineSearch(x, y):

    n, m = x.shape  # for generality
    X0 = np.ones((n, 1))
    x = np.hstack((x, X0))

    w = np.zeros(m + 1)

    for i in range(1000):
        dw = - loss_gradient(x, y, w)
        t = step_length(x, y, w, dw)
        w = w + t * dw

    return w


def step_length(x, y, w, dw, alpha = 0.1, beta = 0.5):
    t = 1
    while loss(x, y, w + t * dw) > loss(x, y, w) + alpha * t * loss_gradient(x, y, w).T @ dw:
        t = beta * t
    return t


def loss(x, y, w):
    return (np.linalg.norm(x @ w - y, 2))**2 / 2


def loss_gradient(x,y,w):
    return x.T @ (x @ w - y)


if __name__ == '__main__':
    x, y = load_data('dataset0.npy')
    w = regression(x, y)
    print(w)
    w = regressionLineSearch(x, y)
    print(w)
    plot(x, y, w)
    
    x, y = load_data('dataset1.npy')
    w = regression(x, y)
    print(w)
    w = regressionLineSearch(x, y)
    print(w)
    plot(x, y, w)

    x, y = load_data('dataset2.npy')
    w = regression(x, y)
    print(w)
    w = regressionLineSearch(x, y)
    print(w)
    plot(x, y, w)

    x, y = load_data('dataset3.npy')
    w = regression(x, y)
    print(w)
    w = regressionLineSearch(x, y)
    print(w)
    plot(x, y, w)

    x, y = load_data('dataset4.npy')
    w = regression(x, y)
    print(w)
    w = regressionLineSearch(x, y)
    print(w)
    print(np.linalg.norm(x @ w[:-1] + w[-1] - y))
