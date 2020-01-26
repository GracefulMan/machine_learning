import numpy as np
from Linear_Model.activation_function import sigmoid


# the gradient of mean square error loss
def mse_gradient(train_x, beta, train_y):
    return 2 * np.dot(train_x.T, np.dot(train_x, beta) - train_y)


# the gradient of mean absolute value error loss
def mae_gradient(train_x, beta, train_y):
    return np.dot(train_x.T,  np.sign(np.dot(train_x, beta) - train_y))


# the gradient of Huber loss
def huber_gradient(train_x, beta, train_y, deta=100):
    if np.mean(np.abs(np.dot(train_x, beta) - train_y)) <= deta:
        return 1 / 2.0 * mse_gradient(train_x, beta, train_y)
    else:
        return deta * mae_gradient(train_x, beta, train_y)


def logistic_gradient_01_response(train_x, beta, train_y):
    return  np.dot( train_x.T, sigmoid(np.dot(train_x, beta)) - train_y)





def gradient_map(name):
    name = name.lower()
    gradient_func = {
        "mse": mse_gradient,
        "mae": mae_gradient,
        "huber": huber_gradient,
        "logistic0": logistic_gradient_01_response
    }
    if name not in gradient_func.keys():
        print("doesn't exist loss function: %s" % name)
        return None
    return gradient_func[name]
