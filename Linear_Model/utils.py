import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from Linear_Model.loss_function import mse
from Linear_Model.activation_function import sigmoid,softmax



# generate dataset.
def generate_dataset(sample_num, features):
    '''
    :param sample_num: indicates the number of dataset's rows.
    :param features:  indicates the number of features
    :return: generated train_data
    '''
    real_beta = np.random.randint(-10, 10, (features, 1)) #generate parameter :beta
    train_x = np.random.random((sample_num, features)) * 10
    train_y = np.dot(train_x, real_beta) + np.random.normal(0, 0.5, (sample_num,1))
    return train_x, train_y, real_beta


# generate dataset for classification
def  generate_classification_dataset(sample_num, class_num, features, negative_sign = -1):
    '''
    :param sample_num: indicates the number of dataset's rows.
    :param class_num:  indicates the  number of classes.
    :param features:  indicates the number of features.
    :param negative_sign: marked negative label as -1(default)
    :return: generated train_data
    '''
    train_x, train_y, real_beta = generate_dataset(sample_num, features)
    train_y = np.sign(train_y)
    if negative_sign == 0:
        train_y[train_y < 0] = 0
    return train_x, train_y, real_beta


# the following function is used to visualize the 1 variable or 2 variables regression.
def visualization(train_x, train_y, beta):
    '''
    this functin is used to visualize the 1 variable or 2 variables regression.
    :param train_x: train data. type: 2 dims numpy array.(m*n)
    :param train_y: label. type : 2 dims numpy array. (m * 1)
    :param beta:
    :return:
    '''
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

def visualization_for_binary_classification(train_x, train_y, beta):
    '''
    :param train_x: train data
    :param train_y: train label
    :return: None
    '''
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    if features != 2:
        print("can't visualize for dim=%s data" % features)
        return None
    fig = plt.figure()
    ax1 = Axes3D(fig)
    mask = (train_y >= 0.5)
    mask = np.concatenate((mask, mask), axis=1)
    first_class_data = train_x[mask].reshape((-1, 2))
    second_class_data = train_x[1 - mask].reshape((-1, 2))
    f_x1 = first_class_data[:, 0]
    f_x2 = first_class_data[:, 1]
    s_x1 = second_class_data[:, 0]
    s_x2 = second_class_data[:, 1]
    ax1.scatter(f_x1, f_x2, color='red')
    ax1.scatter(s_x1, s_x2, color='blue')
    line_x = np.linspace(train_x.min(), train_x.max(), 1000)
    line_y = line_x.copy()
    ax1.plot_surface(line_x, line_y, (line_x * beta[0] + line_y * beta[1]).reshape(-1, 1))
    plt.show()


def show_data_distribution(train_x, train_y, map_shape = (20,20), beta_range =(1,100), vis_loss = False, current_beta=[]):
    '''
    this function is used to visualize the data distribution.
    :param
        train_x, train_y: type : 2 dim numpy array(m,n)train data and corresponding labels.
        map_shape: type :tuple.  the size of graph which will be plotted.
        beta_range: type : tuple. the range of parameter beta.
        vis_loss: type : bool. True: show loss curve on graph.
        current_beta: beta list recorded in training process.
    '''
    m, n = map_shape
    min_value, max_value = beta_range
    x = np.linspace(min_value, max_value, m)
    y = np.linspace(min_value, max_value, n)
    beta1, beta2 = np.meshgrid(x, y)
    beta1_tmp = beta1.reshape((m, n, 1))
    beta2_tmp = beta2.reshape((m, n, 1))
    beta_list = np.concatenate((beta1_tmp, beta2_tmp), axis=2)
    loss = np.zeros(shape=(m, n))
    for i in range(m):
        for j in range(n):
            loss[i, j] = mse(train_y, np.dot(train_x, beta_list[i, j].reshape(2, 1)))

    # plot the data
    fig = plt.figure()
    ax1 = Axes3D(fig)
    if vis_loss:
        current_beta = np.array(current_beta)
        ax2 = fig.gca(projection='3d')
        vis_loss_list = np.zeros(shape=(current_beta.shape[0],))
        for i in range(current_beta.shape[0]):
            vis_loss_list[i] = mse(train_y, np.dot(train_x, current_beta[i]))
        current_beta = current_beta.reshape((-1, 2))
        tmpX = current_beta[:, 0].reshape((-1,))
        tmpY = current_beta[:, 1].reshape((-1,))
        print(tmpX, tmpY,vis_loss_list)
        ax2.plot(tmpX, tmpY, vis_loss_list, color='r', label="loss curve")
        ax2.legend()
    ax1.plot_surface(beta1, beta2, loss,rstride=1, cstride=1, cmap='rainbow')
    plt.show()






