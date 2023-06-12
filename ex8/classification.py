# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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

    
if __name__ == '__main__':

    # Task 1
    print('Dataset R')
    x_train, y_train = load_data('dataset_R_train.npy')
    x_test, y_test = load_data('dataset_R_test.npy')

    ...

    # Task ...
    print('Dataset E')
    x_train, y_train = load_data('dataset_E_train.npy')
    x_test, y_test = load_data('dataset_E_test.npy')

    ...
    
    print('Dataset G')
    x_train, y_train = load_data('dataset_G_train.npy')
    x_test, y_test = load_data('dataset_G_test.npy')

    ...
    
    print('Dataset O')
    x_train, y_train = load_data('dataset_O_train.npy')
    x_test, y_test = load_data('dataset_O_test.npy')

    ...

    print('Dataset digits')
    X, y = load_digits(return_X_y=True)
    plot_numbers(X, y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=66)
