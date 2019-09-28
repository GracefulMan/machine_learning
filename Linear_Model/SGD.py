import numpy as np
from Linear_Model.utils import generate_dataset, visualization
# BGD(batch gradient descent method)

def batch_gradient_descent(train_x, train_y,learning_rate=0.05, iter_num = 10000):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.zeros((features, 1))
    for i in range(iter_num):
        tmp = (1 / sample_nums) * np.dot(train_x.T, np.dot(train_x, beta) - train_y)
        beta -= learning_rate * tmp
    return beta

if __name__ == '__main__':
    train_x, train_y, _ = generate_dataset(100,1)
    beta = batch_gradient_descent(train_x, train_y, learning_rate= 0.005)
    visualization(train_x, train_y, beta)


