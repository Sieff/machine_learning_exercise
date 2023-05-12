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


def polynomial_regression(x, y, Lambda, degree):
    x = generate_poly_matrix(x, degree)

    w = np.linalg.solve((x.T @ x + len(x)*Lambda*np.identity(degree+1)), x.T @ y)
    
    return w


def generate_poly_matrix(x, degree):
    v = np.polynomial.polynomial.polyvander(x, degree)
    return v[:, 0, ::-1]


def regression(x, y):

    # code your solution here
    n, m = x.shape  # for generality
    X0 = np.ones((n, 1))
    x = np.hstack((x, X0))

    w = np.linalg.lstsq(x, y, rcond=None)

    return w[0]


def loss(x, y, w, Lambda, degree):
    x = generate_poly_matrix(x, degree)

    return (np.linalg.norm(x @ w - y, 2))**2 / (2*len(x)) + (Lambda / 2) * np.linalg.norm(w, 2)**2


def kfold_validation(data, Lambda, degree, splits):
    training_errors = np.array([])
    validation_errors = np.array([])
    for i in range(splits):
        x = np.concatenate(np.delete(data, i, axis=0))[:, :-1]
        y = np.concatenate(np.delete(data, i, axis=0))[:, -1]
        w = polynomial_regression(x, y, Lambda, degree)

        training_loss = loss(x, y, w, Lambda, degree)
        training_errors = np.append(training_errors, training_loss)

        x = data[i][:, :-1]
        y = data[i][:, -1]
        validation_loss = loss(x, y, w, Lambda, degree)
        validation_errors = np.append(validation_errors, validation_loss)
    training_loss = np.mean(training_errors)
    validation_loss = np.mean(validation_errors)
    return training_loss, validation_loss


def shuffle(x, y):
        t = np.hstack((x, y.reshape((y.shape[0], 1))))
        t_shuffed = np.random.permutation(t)
        y_shuffled = t_shuffed[:, -1]
        x_shuffled = t_shuffed[:, :-1]

        return x_shuffled, y_shuffled


def plot_loss(training_losses, validation_losses):
    scale = np.arange(100)
    plt.plot(scale, np.flip(training_losses), color='red', label='training')
    plt.plot(scale, np.flip(validation_losses), color='blue', label='validation')
    plt.legend()

    plt.tight_layout()
    plt.savefig('fig.pdf')
    plt.show()


def test_lambda(Lambda, degree):
    data = np.load('dataset_poly_test.npy')
    x = data[:, :-1]
    y = data[:, -1]
    w = polynomial_regression(x, y, Lambda, degree)
    plot(x, y, w)
    print("Lambda:")
    print(Lambda)
    print("Test loss:")
    print(loss(x, y, w, Lambda, degree))


if __name__ == '__main__':
    degree = 6
    splits = 10
    data = np.load('dataset_poly_train.npy')
    np.random.shuffle(data)
    data = np.split(data, splits)

    lambdas = np.logspace(-9, 3, 100)

    training_losses = []
    validation_losses = []
    for Lambda in lambdas:
        training_loss, validation_loss = kfold_validation(data, Lambda, degree, splits)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    plot_loss(training_losses, validation_losses)
    best_lambda = lambdas[np.argmin(validation_losses)]

    test_lambda(best_lambda, degree)


