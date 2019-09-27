import numpy as np
from numpy.linalg.linalg import pinv
from linear_regression.visuaize import visualization
# generate dataset.
def generate_dataset(sample_num, features):
    '''
    :param sample_num: indicates the number of dataset's rows.
    :param features:  indicates the number of features
    :return: generated train_data
    '''

    real_beta = np.random.randint(1, 10, (features, 1)) #generate parameter :beta
    train_x = np.random.random((sample_num, features)) * 10
    #解释一下
    train_y = np.dot(train_x, real_beta) + np.random.normal(0, 0.5, (sample_num,1))
    return train_x, train_y


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

    train_x, train_y = generate_dataset(sample_num, features)
    beta = least_squre(train_x, train_y)
    print("estimated parameter:", beta)
    visualization(train_x, train_y, beta)