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

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def create_random_features(num):
    return np.random.uniform(low=-1, high=1, 
                            size=[num, 4, 4, 8])

# define models

# 32x32 [-1,1] image
discriminator = Sequential([
    Convolution2D(32, 5, 5, border_mode='same', subsample=(2, 2), input_shape=[32, 32, 1]),
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
    ELU(),
    UpSampling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    BatchNormalization(),
    ELU(),
    UpSampling2D(), #16x16
    Convolution2D(128, 3, 3, border_mode='same'),
    BatchNormalization(),
    ELU(),
    UpSampling2D(), # 32x32
    Convolution2D(1, 5, 5, border_mode='same', activation='tanh')
], name="generator")

# setup models

print("setup discriminator")
set_trainable(discriminator, True)
opt_d = Adam(lr=0.000005, beta_1=0.1)
discriminator.compile(optimizer=opt_d, 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

print("setup dcgan")
set_trainable(discriminator, False)
dcgan = Sequential([generator, discriminator])
opt_g = Adam(lr=0.0002, beta_1=0.5)
dcgan.compile(optimizer=opt_g, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])




# training

X_train = np.load(npy_path)

def evaluate(generate_batch_num = 1000):
    random_features = create_random_features(generate_batch_num)
    pred = dcgan.predict(random_features)
    faked = np.sum(pred>0.5)
    return faked

batch_size = 100
wait = 0
faked_curve = np.array([])
for epoch in range(1, sys.maxsize):

    print("epoch: {0}".format(epoch))
    
    np.random.shuffle(X_train)
    rnd = create_random_features(len(X_train))

    val_interval = 50
    rnd_val = create_random_features(1000)
    y_rnd_val = [0]*len(rnd_val)

    # train on batch
    for i in range(math.ceil(len(X_train)/batch_size)):
        print("batch:", i, end='\r')
        X_batch = X_train[i*batch_size:(i+1)*batch_size]
        rnd_batch = rnd[i*batch_size:(i+1)*batch_size]

        generated = generator.predict(rnd_batch)
        y_batch = [0]*len(X_batch) + [1]*len(generated)
        X_batch = np.append(X_batch, generated, axis=0)
        _, ad = discriminator.train_on_batch(X_batch, y_batch)

        if i%val_interval == 0:
            print("")
            generated = generator.predict(rnd_val)
            loss_d, acc_d = discriminator.evaluate(generated, [1]*len(generated))

        _, ag = dcgan.train_on_batch(rnd_batch, [0]*len(rnd_batch))

        if i%val_interval == 0:
            loss_g, acc_g = dcgan.evaluate(rnd_val, y_rnd_val)
            print("")
            print("dis: loss: {0:.5f} acc: {1:.3f}".format(loss_d, acc_d))
            print("gen: loss: {0:.5f} acc: {1:.3f}".format(loss_g, acc_g))
    
    # output
    test_num = 1000
    rnd_test = create_random_features(test_num)
    _, faked = dcgan.evaluate(rnd_test, [0]*test_num)
    print("")
    print("epoch end:", ad, ag)
    print("faked: {0}".format(faked))

    np.append(faked_curve, faked)
    np.save(os.path.join(files_dir, "./faked_curve"), faked_curve)

    # save model
    if epoch%50 == 0:
        # avoiding bug
        # https://github.com/fchollet/keras/pull/4338
        model = Sequential([generator, discriminator])
        model.save(os.path.join(files_dir, "./models/{0}.h5".format(epoch)))
        print("Save: {0}.h5".format(epoch))
    print("")

    if acc_d==0 or faked==0:
        wait += 1
        if wait>100:
            print("wait reach 100")
            exit(0)
    else:
        wait = 0
