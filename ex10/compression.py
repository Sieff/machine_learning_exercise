import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X = np.load('./box.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    ...
    
    X = np.load('./spring_1.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    ...
    
    X = np.load('./spring_2.npy')
    plt.figure().add_subplot(111, projection = '3d').plot(*X.T, 'o')
    plt.show()

    ...
