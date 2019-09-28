import numpy as np
from numpy.linalg.linalg import pinv
from Linear_Model.utils import visualization, generate_dataset

# least square algorithm
def least_squre(train_x, train_y):
    '''
    :param train_x: train sample
    :param train_y: train label
    :return: parameter beta.
    '''
    beta = pinv(np.dot(train_x.T, train_x)).dot(train_x.T).dot(train_y)
    return beta


if __name__ == '__main__':
    sample_num = 20
    features = 1

    train_x, train_y, _ = generate_dataset(sample_num, features)
    beta = least_squre(train_x, train_y)
    print("estimated parameter:", beta)
    visualization(train_x, train_y, beta)