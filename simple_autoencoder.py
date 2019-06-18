import numpy as np
from keras.layers import Dense,Input,Conv2D,Flatten,BatchNormalization,MaxPooling2D,UpSampling2D,Deconv2D,Conv2DTranspose
from keras.datasets import mnist,cifar10
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
#parameters
EPOCHS = 10
INIT_LR = 1e-3
BS = 128
pic_size = 28
dim_ = 1
# load data

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.  
x_test = x_test.astype('float32') / 255.  
x_train = np.reshape(x_train, (len(x_train), pic_size, pic_size, dim_))
x_test = np.reshape(x_test, (len(x_test), pic_size, pic_size, dim_))
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)   
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)   
x_train_noisy = np.clip(x_train_noisy, 0., 1.)  #[0.,1.]
x_test_noisy = np.clip(x_test_noisy, 0., 1.)  
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
# construct Model

input_img = Input(shape=(pic_size,pic_size,dim_))
x = Conv2D(32,(3,3),padding='same',activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Conv2D(32, (3, 3), padding="same",activation='relu')(x)
encoder = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Conv2D(32, (3, 3),padding='same', strides=(2, 2),activation='relu')(encoder)
x = UpSampling2D(size=(2, 2))(x)
decoder = Conv2DTranspose(3, (3, 3),padding='same', strides=(2, 2),activation='relu')(x)
decoder = Conv2DTranspose(3, (3, 3),padding='same', strides=(2, 2),activation='relu')(decoder)
decoder = BatchNormalization()(decoder)
decoder = Flatten()(decoder)
decoder = Dense(pic_size*pic_size*dim_,activation='sigmoid')(decoder)

autoencoder=Model(inputs=input_img,outputs=decoder)
optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss='binary_crossentropy', optimizer=optimizer)


autoencoder.summary()
plot_model(autoencoder,to_file='./model.png',show_shapes=True)

autoencoder.fit(x_train_noisy, x_train, epochs=EPOCHS, batch_size=BS,
                shuffle=True, validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='autoencoder', write_graph=False)])

#model.fit(x_train_nosiy,x_train,epochs=EPOCHS,batch_size=BS,verbose=1)
decoder_img = autoencoder.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #noisy data
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].reshape(pic_size,pic_size))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #predict
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoder_img[i].reshape(pic_size, pic_size))
    #plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # original
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(pic_size, pic_size))
    #plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.show()