#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
import numpy as np

files_dir = os.path.dirname(__file__)
npy_dir = os.path.join(files_dir, "./dae_train_data")
model_path = os.path.join(files_dir, "./dae_model.h5")

encoder = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[32, 32, 1]),
    ELU(),
    MaxPooling2D(border_mode='same'), #16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    MaxPooling2D(border_mode='same'), #8x8
    Convolution2D(8, 3, 3, border_mode='same'),
    ELU(),
    MaxPooling2D(border_mode='same'), #4x4
], name="encoder")

decoder = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[4, 4, 8]),
    ELU(),
    UpSampling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    UpSampling2D(), # 16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    UpSampling2D(), # 32x32
    Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid'),
], name="decoder")

dae = Sequential([encoder, decoder])

dae.compile(optimizer='adam', loss='binary_crossentropy')

def add_noise(images, noise_ratio=0.2):
    noise = np.random.uniform(images.shape)
    noised_images = np.copy(images)
    noised_images[noise < noise_ratio] = 1
    return noised_images

trains = np.load(os.path.join(npy_dir, "./train.npy"))
vals = np.load(os.path.join(npy_dir, "./val.npy"))

y_train = np.concatenate([trains]*3, axis=0)
y_val = np.concatenate([vals]*3, axis=0)
X_train = np.concatenate([
    add_noise(trains, 0.1), 
    add_noise(trains, 0.15), 
    add_noise(trains, 0.2)
], axis=0)
X_val = np.concatenate([
    add_noise(vals, 0.1), 
    add_noise(vals, 0.15), 
    add_noise(vals, 0.2)
], axis=0)

dae.fit(X_train, y_train,
       nb_epoch=500,
       validation_data=(X_val, y_val),
       callbacks=[
           EarlyStopping(patience=2),
           TensorBoard(log_dir='/tmp/abyss2_log/dae'),
           ModelCheckpoint(model_path, save_best_only=True)
       ])