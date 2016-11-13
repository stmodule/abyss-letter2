#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Reshape
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, Callback
import numpy as np
import os, math, sys

files_dir = os.path.dirname(__file__)
npy_path = os.path.join(files_dir, "./dcgan_train_data.npy")
model_path = os.path.join(files_dir, "./dcgan_model.h5")

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def create_random_features(num):
    return np.random.uniform(low=-1, high=1, 
                            size=[num, 4, 4, 8])


# 32x32 [-1,1] image
discriminator = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), input_shape=[32, 32, 1]),
    LeakyReLU(), # [16, 16, 32]
    Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2)),
    LeakyReLU(), # [8, 8, 64]
    Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)),
    LeakyReLU(), # [4, 4, 64]
    Flatten(),
    Dense(1024),
    LeakyReLU(),
    Dense(1, activation='sigmoid')
], name="discriminator")

# input shape=[4, 4, 8] range=[-1, 1]
generator = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[4, 4, 8]),
    BatchNormalization(),
    Activation('relu'),
    UpSampling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    UpSampling2D(), #16x16
    Convolution2D(128, 3, 3, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    UpSampling2D(), # 32x32
    Convolution2D(1, 3, 3, border_mode='same', activation='tanh')
], name="generator")

print("setup discriminator")
opt_d = Adam(lr=0.0001, beta_1=0.5)
discriminator.compile(optimizer=opt_d, 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

print("setup dcgan")
set_trainable(discriminator, False)
dcgan = Sequential([generator, discriminator])
opt_g = Adam(lr=0.001, beta_1=0.5)
dcgan.compile(optimizer=opt_g, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


abyss_letters = np.load(npy_path)

def evaluate(generate_batch_num = 1000):
    random_features = create_random_features(generate_batch_num)
    pred = dcgan.predict(random_features)
    faked = np.sum(pred>0.5)
    return faked


batch_size = 128
wait = 0
for epoch in range(sys.maxsize):
    
    generated = generator.predict(create_random_features(len(abyss_letters)))

    X_train = np.append(abyss_letters, generated, axis=0)
    y_train = np.append(np.ones(len(abyss_letters)), np.zeros(len(generated)))
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    rnd = create_random_features(len(X_train))

    for i in range(math.ceil(len(X_train)/batch_size)):
        print("batch:", i, end='\r')
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train[i*batch_size:(i+1)*batch_size]
        rnd_batch = rnd[i*batch_size:(i+1)*batch_size]

        loss_d, acc_d = discriminator.train_on_batch(X_batch, y_batch)

        loss_g, acc_g = dcgan.train_on_batch(rnd_batch, np.ones(len(rnd_batch)))
    
    test_num = 1000
    faked = evaluate(1000)
    print("epoch: {0}                    ".format(epoch))
    print("loss_d: {0:e} acc_d {1:.3f}".format(loss_d, acc_d))
    print("loss_g: {0:e} acc_g {1:.3f}".format(loss_g, acc_g))
    print("faked: {0}/{1}".format(faked, test_num))

    if acc_d==0 or faked==0:
        wait += 1
        if wait>100:
            print("waits reach 100")
            exit(0)
    else:
        wait = 0

    # save model
    if epoch%50 == 0:
        # avoiding bug
        # https://github.com/fchollet/keras/pull/4338
        model = Sequential([generator, discriminator])
        model.save(os.path.join(files_dir, "./models/{0}.h5".format(epoch)))
        print("Save: {0}.h5".format(epoch))
    print("")
