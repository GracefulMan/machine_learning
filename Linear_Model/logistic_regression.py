import sys
sys.path.append("/Users/mahongying/Desktop/machine_learning/")
import numpy as np
from Linear_Model.utils import generate_classification_dataset,visualization_for_binary_classification
from Linear_Model.activation_function import sigmoid
from Linear_Model.loss_function import loss_map
from data_processing import data_normalization
from Linear_Model.loss_function_gradient import gradient_map
def sgd_with_mini_batch(train_x, train_y, learning_rate = 5e-2, batch_size = 32, loss='logistic0', iter_num=10000, show_loss=False):
    sample_nums = train_x.shape[0]
    features = train_x.shape[1]
    beta = np.random.normal(loc=0.0, scale= 1.0,size=(features, 1))
    print(beta.shape)
    for i in range(iter_num):
        # sample the partial data from train data.
        '''
        np.random.choice(a,size,replace,p)
        :param a:the list which the elements will be selected.
        :param size: the number of elements will be chosen.
        :param replace: True: after choosing one element, put back to element to "a",it means this element has probability
        to be chosen by next time.
        :param type(list), the probability of choosing, default is uniform distribution.
        '''
        for j in range(sample_nums // batch_size):
            index = np.random.choice(range(sample_nums),batch_size, replace=False)
            batch_x = train_x[index]
            batch_y = train_y[index]
            beta -= learning_rate * gradient_map(loss)(batch_x, beta, batch_y)
        if show_loss and i % 10==0:
            print("iter:%s\t\tloss:%s" % (i, loss_map(loss)(train_y, np.dot(train_x, beta))))
    return beta


if __name__ == '__main__':
    train_x, train_y, _ = generate_classification_dataset(2000, 2, 2, negative_sign=0)
    train_x = data_normalization.minMaxScaler(train_x)
    beta = sgd_with_mini_batch(train_x, train_y,learning_rate=0.01, batch_size=64,show_loss=True)
    visualization_for_binary_classification(train_x, train_y, beta)
