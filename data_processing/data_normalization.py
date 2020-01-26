import numpy as np


def minMaxScaler(data, axis=2):
    '''
    :param data: the data which need to be normalized.
    :param axis: the direction of normalization.
        default:2 (all)
        0: row
        1:column
    :return: processed data.(numpy array)
    '''
    # use min-max normalization for row.
    data = np.array(data, dtype=np.float32)
    if axis == 0:
        for i in range(data.shape[0]):
            data[i, ] = (data[i, ] - data[i, ].min()) / (data[i, ].max() - data[i, ].min())
    # use min-max normalization for column.
    elif axis == 1:
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())
    # use min-max normalization for full data.
    elif axis == 2:
        data = (data - data.min()) / (data.max() - data.min())
    return data


def zScoreNormalization(data, axis=2):
    '''
    :param data: the data which need to be normalized.
    :param axis: the direction of normalization.
    :return: processed data.(numpy array)
    '''
    # use z-score normalization for row.
    data = np.array(data, dtype=np.float32)
    if axis == 0:
        for i in range(data.shape[0]):
            data[i, ] = (data[i, ] - data[i, ].mean()) / data[i, ].std()
    # use z-score normalization for column.
    elif axis == 1:
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - data[:, i].mean()) / data[:, i].std()
    # use z-score normalization for full data.
    elif axis == 2:
        data = (data - data.mean()) / data.std()
    return data


