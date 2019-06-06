import numpy as np
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D,Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.datasets import mnist, cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_conv = 8
batch_size = 100
intermediate_dim = 512
epochs = 25
pic_size = x_train.shape[1]
latent_dim = pic_size * pic_size
epsilon_std = 1.0
dim_ = 3
x_train = np.reshape(x_train, [-1, pic_size, pic_size, dim_])
x_test = np.reshape(x_test, [-1, pic_size, pic_size, dim_])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
#np.save('x_test.npy',x_test)
print(x_train.shape,x_test.shape)
x = Input(batch_shape=(batch_size,pic_size, pic_size,dim_))
conv_1 = Reshape((pic_size,pic_size,dim_))(x)
conv_1 = Conv2D(1, kernel_size=num_conv,padding='same', activation='relu')(conv_1)
conv_2 = Conv2D(64, kernel_size=num_conv,padding='same', strides=2, activation='relu')(conv_1)
conv_3 = Conv2D(64, kernel_size=num_conv,padding='same', activation='relu')(conv_2)
flatten = Flatten()(conv_3)
hidden = Dense(intermediate_dim, activation='relu')(flatten)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., stddev=epsilon_std)
    return(z_mean + K.exp(z_log_var/2) * epsilon)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(256, activation='relu')(z)
decoder = Dense(pic_size * pic_size*dim_, activation='relu')(decoder_h)
print(decoder.shape)
decoder = Reshape((pic_size, pic_size,dim_))(decoder)
de_conv_1 = Conv2DTranspose(64, kernel_size=num_conv,padding='same', activation='relu')(decoder)
de_conv_2 = Conv2DTranspose(64, kernel_size=num_conv,padding='same', activation='relu')(de_conv_1)
#upsamp = UpSampling2D(2)(de_conv_2)
x_decoded_mean = Conv2DTranspose(3, kernel_size=num_conv,
                        padding='same', activation='relu')(de_conv_2)
print(x_decoded_mean.shape)
x_decoded_mean = Reshape([pic_size, pic_size,dim_] )(x_decoded_mean)

def vae_loss(x, x_decoded_mean,loss_type = 'mse'):
    if loss_type == 'mse':
        reconstruction_loss = mse(K.flatten(x), K.flatten(x_decoded_mean))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(x),
                                                  K.flatten(x_decoded_mean))
    reconstruction_loss *= pic_size * pic_size
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) -K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)
vae = Model(x, x_decoded_mean)
vae.summary()
vae.add_loss(vae_loss(x,x_decoded_mean,loss_type='corss'))
vae.compile(optimizer='rmsprop')
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))
vae.save('myvae.h5')
res = vae.predict(x_test,batch_size=batch_size)
np.save('test2.npy',res)
