import numpy as np
from keras.layers import Dense,Input
from keras.datasets import mnist,cifar10
from keras.models import Model
import matplotlib.pyplot as plt
#parameters
EPOCHS = 20
BS = 128
# load data
(train_x, _) ,(test_x, _) = mnist.load_data()
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype("float32") / 255.0
train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0],-1)

# add random noise in order to prevent over-fitting
#add random noise
x_train_nosiy = train_x + 0.3 * np.random.normal(loc=0., scale=1., size=train_x.shape)
x_test_nosiy = test_x + 0.3 * np.random.normal(loc=0, scale=1, size=test_x.shape)
x_train_nosiy = np.clip(x_train_nosiy, 0., 1.)
x_test_nosiy = np.clip(x_test_nosiy, 0, 1.)
print(x_train_nosiy.shape, x_test_nosiy[0].shape)

# construct Model
input_img = Input(shape=(x_test_nosiy.shape[1],))
encoder = Dense(500, activation='relu')(input_img)
decoder = Dense(x_test_nosiy.shape[1],activation='sigmoid')(encoder)
model = Model(inputs=input_img,outputs=decoder)
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(x_train_nosiy,train_x,
          epochs=EPOCHS,
          batch_size=BS,
          verbose=1,
          validation_data=(test_x,test_x)
          )
decoder_img = model.predict(x_test_nosiy)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #noisy data
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test_nosiy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #predict
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoder_img[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # original
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(test_x[i].reshape(28, 28))
    plt.gray()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

plt.show()