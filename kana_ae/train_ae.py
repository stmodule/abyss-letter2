#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
import numpy as np

files_dir = os.path.dirname(__file__)
npy_dir = os.path.join(files_dir, "./ae_train_data")
model_path = os.path.join(files_dir, "./ae_model.h5")

feature_filters = 8

encoder = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[32, 32, 1]),
    ELU(),
    MaxPooling2D(border_mode='same'), #16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    MaxPooling2D(border_mode='same'), #8x8
    Convolution2D(feature_filters, 3, 3, border_mode='same'),
    ELU(),
    MaxPooling2D(border_mode='same'), #4x4
])

decoder = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[4, 4, feature_filters]),
    ELU(),
    UpSampling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    UpSampling2D(), # 16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    ELU(),
    UpSampling2D(), # 32x32
    Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid'),
])

ae = Sequential([encoder, decoder])

ae.compile(optimizer='adam', loss='binary_crossentropy')

trains = np.load(os.path.join(npy_dir, "./train.npy"))
vals = np.load(os.path.join(npy_dir, "./val.npy"))

ae.fit(trains, trains,
       nb_epoch=100,
       validation_data=(vals, vals),
       callbacks=[
           EarlyStopping(patience=2),
           TensorBoard(log_dir='/tmp/abyss2_log/ae'),
           ModelCheckpoint(model_path, save_best_only=True)
       ])