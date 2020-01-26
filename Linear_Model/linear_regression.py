import numpy as np
import sys

sys.path.append("/Users/mahongying/Desktop/machine_learning/")
from Linear_Model.utils import generate_dataset, visualization, show_data_distribution
from Linear_Model.loss_function import mse, loss_map
from Linear_Model.loss_function_gradient import gradient_map
params = []

'''
the following functions omit the error term b(epision)
'''


# BGD(batch gradient descent method)
def batch_gradient_descent(train_x, train_y, learning_rate=0.05, iter_num=10000, show_loss=False):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.zeros((features, 1))  # y = X\beta + epision
    for i in range(iter_num):
        tmp = (1 / sample_nums) * np.dot(train_x.T, np.dot(train_x, beta) - train_y)
        beta -= learning_rate * tmp
        params.append(beta)
        if show_loss:
            print(mse(train_y, np.dot(train_x, beta)))
    return beta


# stochastic gradient descent without mini-batch.(each sample updates parmeters )
def sgd(train_x, train_y, learning_rate=0.05, iter_num=10000, show_loss=False):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.random.normal(size=(features, 1))
    print(train_x.shape, beta.shape,train_y.shape)
    for i in range(iter_num):
        for j in range(sample_nums):
            '''the reason for reshape:
                numpy doesn't support column or row vector.
            '''
            tmp = np.dot(train_x[j, :].reshape((1, features)).T, np.dot(train_x[j, :].reshape((1, features)), beta) - train_y[j])
            beta -= learning_rate * tmp
        if show_loss:
            print(mse(train_y, np.dot(train_x, beta)))
    return beta


# bug bug bug!!!
def sgd_with_mini_batch(train_x, train_y, learning_rate=0.0001, iter_num=1000, batch_size=32):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.random.normal(size=(features, 1))
    for i in range(iter_num):
        # fetch data from dataset in fixed batch_size.
        for j in range(sample_nums // batch_size):
            index = np.random.choice(range(sample_nums), batch_size, replace=False)
            batch_x = train_x[index]
            batch_y = train_y[index]
            tmp = (1 / batch_size) * np.dot(batch_x.T, np.dot(batch_x, beta) - batch_y)
            beta -= learning_rate * tmp
    return beta


def sgd_with_mini_batch_diff_loss(train_x, train_y, learning_rate=1e-5, loss='mse', iter_num=10000, batch_size=32,
                                  show_loss=False):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.random.normal(size=(features, 1))
    for i in range(iter_num):
        # fetch data from dataset in fixed batch_size.
        for j in range(sample_nums // batch_size):
            index = np.random.choice(range(sample_nums), batch_size, replace=False)
            batch_x = train_x[index]
            batch_y = train_y[index]
            tmp = (1 / batch_size) * gradient_map(loss)(batch_x, beta, batch_y)
            beta -= learning_rate * tmp
        if show_loss and i % 50 == 0:
            print("iter:%s\t\tloss:%s" % (i, loss_map(loss)(np.dot(train_x, beta), train_y)))
    return beta


# least squares
def least_squares(train_x, train_y):
    # \bar{\beta} = arg min_\beta |Y-X\beta|^2
    return np.linalg.pinv(np.dot(train_x.T, train_x)).dot(train_x.T).dot(train_y)


if __name__ == '__main__':
    train_x, train_y, _ = generate_dataset(100, 10)
    #beta = batch_gradient_descent(train_x, train_y, learning_rate= 0.005, show_loss=True)
    # print(beta)
    # show_data_distribution(train_x, train_y,map_shape=(50, 50),beta_range=(0, 10),vis_loss=True,current_beta=params)
    #beta1 = sgd(train_x, train_y, learning_rate=1e-5, show_loss= True)
    # print(beta1)
    # beta2 = batch_gradient_descent(train_x,train_y, learning_rate= 0.005)
    # beta3 = sgd(train_x, train_y, learning_rate = 1e-6)
    # beta4 = sgd_with_mini_batch(train_x, train_y, learning_rate=1e-5)
    # print(beta4)
    # visualization(train_x, train_y, beta4)
    beta5 = sgd_with_mini_batch_diff_loss(train_x, train_y, loss='huber', show_loss=True, iter_num=10000)
    # beta6 = least_squares(train_x, train_y)
    #visualization(train_x, train_y, beta5)
    # print(beta6)
    # print(_)
