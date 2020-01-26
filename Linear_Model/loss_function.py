# this file implements the commonly used loss funcion
import numpy as np
from Linear_Model.activation_function import sigmoid
# mse, mas and huber are used for robust linear regression.

def mse(y_true, y_pred):
    # mean squre error .
    return np.mean(np.square(y_pred - y_true))


def mae(y_true, y_pred):
    # mean absolute value loss
    return np.mean(np.abs(y_true - y_pred))


def huber(y_true, y_pred, deta = 100):
    # huber loss
    # deta is a cut-off value that is heuristically chosen, so that beyond deta we penalize the deviation by absolute value.
    if mae(y_true, y_pred) <= deta:
        return 1 / 2.0 * mse(y_true, y_pred)
    else:
        return deta * mae(y_true, y_pred) - np.square(deta) / 2


def logistic_for_0_1_responses(y_true, y_pred):
    # label belongs to 0 and 1
    return -np.sum(y_true * np.log(sigmoid(y_pred) + 1e-10) + (1 - y_true) * np.log(1 - sigmoid(y_pred) + 1e-10))


def logistic_loss(y_true, y_pred):
    # label belongs to -1 and 1
    return np.sum(np.log(1 + np.exp(-y_true * y_pred)))


def loss_map(name):
    name = name.lower()
    loss_func ={
        "mse": mse,
        "huber": huber,
        "mae": mae,
        "logistic":logistic_loss,
        "logistic0":logistic_for_0_1_responses
    }

    if name not in loss_func.keys():
        print("doesn't exist loss function: %s"%name)
        return None
    return loss_func[name]
