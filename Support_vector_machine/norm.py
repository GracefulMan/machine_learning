'''
this file implements the L-k norm
'''

import numpy as np
def Norm(data, k = 2):
    '''
    :param data: the data which will be implemented norm ops.
    :param k: L-k norm, default: 2
    :return: processed data.(type:float32)
    '''
    if k == 0:
        return np.sum(data != 0)
    root = 1. / k
    return np.power(np.sum(np.abs(data) ** k), root)


if __name__ == '__main__':
    a = np.array([-1, 1, 3])
    for i in range(10):
        print('norm %d:%f'% (i, Norm(a, i)))
