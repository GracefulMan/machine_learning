import numpy as np
from Linear_Model.utils import generate_classification_dataset


def fisher_model(sample, label):
    index = (label > 1)
    tmp = index
    for i in range(sample.shape[1] - 1):
        tmp = np.concatenate((tmp, index), axis=1)
    print(tmp.shape)
    print(sample.shape)
    x0 = sample[tmp]
    x1 = sample[tmp == False]
    sample_num, feature_num = np.shape(sample)
    print(np.mean(x0))
    u0 = np.mean(x0, axis=0).reshape(1, feature_num)
    u1 = np.mean(x1, axis=0).reshape(1, feature_num)
    sw = (x1 - u1).T.dot((x1 - u1))+(x0 - u0).T.dot((x0 - u0))
    w = (u0 - u1).dot(np.linalg.inv(sw))
    #计算类间散度和类内散度
    res1 = x0.dot(w.T)
    m1 = np.mean(res1,axis=0)[0]
    sum1 = 0
    for i in range(len(res1)):
        sum1 += (res1[i][0] - m1)**2
    res2 = x1.dot(w.T)
    m2 = np.mean(res2, axis=0)[0]
    sum2 = 0
    for i in range(len(res2)):
        sum2 += (res2[i][0] - m2)**2
    #类间散度
    print("intra-class varience:", sum2 + sum1)
    print("inter-class varience:", (m1 - m2)**2)
    return w


def split_dataset(data, label, rate = 0.9):
    sample_num = np.int(data.shape[0] * rate)
    return data[:sample_num], label[:sample_num], data[sample_num:], label[sample_num:]


if __name__ == '__main__':
    train_x, train_y, _= generate_classification_dataset(1000, 2, 10, negative_sign= 0)
    train_x, train_y, test_x, test_y = split_dataset(train_x, train_y, 0.5)
    w = fisher_model(train_x, train_y)
    print(w)


    

