
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist



noise_factor = 0.05
def load_data():
    train_data = np.load('dog_train.npy')
    test_data = np.load('dog_test.npy')
    return train_data[:1000,:,:],test_data[:1000,:,:]
x_train , x_test = load_data()
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0, 1.)


image_size = x_train.shape[1]
# 网络参数
input_shape = (image_size, image_size, 3)
batch_size = 100
kernel_size = 3
filters = 16
latent_dim = 256 # 隐变量取2维只是为了方便后面画图
epochs = 30


x_in = Input(shape=input_shape)
x = x_in

for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# 备份当前shape，等下构建decoder的时候要用
shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(16, activation='relu')(x)
# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
# 先搭建为一个独立的模型，然后再调用模型
latent_inputs = Input(shape=(latent_dim,))
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same')(x)

# 搭建为一个独立的模型
decoder = Model(latent_inputs, outputs)

x_out = decoder(z)

# 建立模型
vae = Model(x_in, x_out)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.binary_crossentropy(x_in, x_out), axis=[1, 2, 3])
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit(x_train_noisy,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_noisy, None))


# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x_in, z_mean)
x_test_encoded = vae.predict(x_test_noisy, batch_size=batch_size)

np.save('gg.npy',x_test_encoded[:20])