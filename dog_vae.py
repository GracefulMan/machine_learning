import numpy as np
from skimage import io
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D,Conv2DTranspose,add,BatchNormalization,MaxPooling2D
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy
#from keras.utils import plot_model
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


num_conv = 8
batch_size = 50
intermediate_dim = 64
epochs = 50
pic_size = x_train.shape[1]
epsilon_std = 1.0
dim_ = 3
latent_dim =  256
print(x_train.shape,x_test.shape)
x = Input(batch_shape=(batch_size,pic_size, pic_size,dim_))
#conv_1 = Reshape((pic_size,pic_size,dim_))(x)
conv_1 = Conv2D(64, kernel_size=(16,16),padding='same', activation='relu')(x)
conv_2 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_3 = Conv2D(128,kernel_size=(8,8),padding='same',activation='relu')(conv_2)
conv_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
conv_3 = Conv2D(256,kernel_size=(4,4),padding='same')(conv_3)
#conv_3 = Conv2D(128, kernel_size=num_conv,padding='same', activation='relu')(conv_2)
#conv_2 = Conv2D(64, kernel_size=num_conv,padding='same', strides=2, activation='relu')(conv_2)
#conv_3 = Conv2D(3, kernel_size=num_conv,padding='same', activation='relu')(conv_2)
#conv_3 =BatchNormalization()(conv_3)
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
decoder_h = Dense(latent_dim, activation='relu')(z)
decoder = Dense(latent_dim, activation='relu')(decoder_h)
print(decoder.shape)
decoder = Reshape((16, 16,-1))(decoder)
print(decoder.shape)
de_conv_1 = Conv2DTranspose(8, kernel_size=(4, 4),padding='same', activation='relu')(decoder)
de_conv_1 = UpSampling2D(2)(de_conv_1)
de_conv_2 = Conv2DTranspose(32, kernel_size=(8, 8),padding='same', activation='relu')(de_conv_1)
de_conv_2 = UpSampling2D(2)(de_conv_2)
x_decoded_mean = Conv2DTranspose(64, kernel_size=(16, 16),
                        padding='same', activation='relu')(de_conv_2)
x_decoded_mean = UpSampling2D(2)(x_decoded_mean)
x_decoded_mean = Conv2DTranspose(3, kernel_size=(4, 4),padding='same',activation='relu')(x_decoded_mean)

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
#plot_model(vae, to_file='my_vae_cnn.png', show_shapes=True)
vae.summary()
vae.add_loss(vae_loss(x,x_decoded_mean,loss_type=''))
vae.compile(optimizer='adam')
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test_noisy, None))
vae.save('vae_dog.h5')
res = vae.predict(x_test_noisy,batch_size=batch_size)
np.save('test2.npy',res[:20,])
