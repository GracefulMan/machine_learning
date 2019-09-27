import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# the following function is used to visualize the 1 variable or 2 variables regression.
def visualization(train_x, train_y,beta):
    if np.shape(beta)[0] > 2:
        print('exceed dim.')
        return 0
    if np.shape(beta)[0] == 1:
        beta = beta[0][0]
        x = np.linspace(np.min(train_x), np.max(train_x), 50)
        y = beta * x
        plt.figure()
        plt.plot(x, y, color='red', linewidth = 1.0, label = 'regression line')
        plt.scatter(train_x, train_y, color='blue',label='real data')
        plt.legend()
        plt.show()
    else:
        beta1, beta2 = beta[0][0], beta[1][0]
        x1 = np.linspace(np.min(train_x[:, 0]), np.max(train_x[:, 0]), 50)
        x2 = np.linspace(np.min(train_x[:, 1]), np.max(train_x[:, 1]), 50)
        x1, x2 = np.meshgrid(x1, x2)
        y = beta1 * x1 + beta2 * x2
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(train_x[:, 0], train_x[:, 1], train_y, color='r', label = 'train data')
        ax.plot_surface(x1, x2, y,label = 'regression surface')
        plt.show()