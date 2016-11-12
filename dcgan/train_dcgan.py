#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, Callback
import numpy as np
import os

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

class GeneratorEarlyStopping(Callback):
    def __init__(self, nb_vals, percentage=0.95):
        super(GeneratorEarlyStopping, self).__init__()
        self.nb_vals = nb_vals
        self.percentage = percentage

    def on_epoch_end(self, epoch, logs={}):
        features = create_random_features(self.nb_vals)
        pred = self.model.predict(features)
        pred = pred > 0.5

        correct = np.sum(pred)
        
        if correct > self.nb_vals*(1-self.percentage):
            self.model.stop_training = True

discriminator = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[32, 32, 1]),
    LeakyReLU(),
    MaxPooling2D(), # 16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    LeakyReLU(),
    MaxPooling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    LeakyReLU(),
    MaxPooling2D(), # 4x4
    Flatten(),
    Dense(512),
    LeakyReLU(),
    Dense(1, activation='sigmoid')
], name="discriminator")

# [4, 4, 8] -1 to 1 elems
generator = Sequential([
    Convolution2D(32, 3, 3, border_mode='same', input_shape=[4, 4, 8]),
    BatchNormalization(),
    ELU(),
    UpSampling2D(), # 8x8
    Convolution2D(64, 3, 3, border_mode='same'),
    BatchNormalization(),
    ELU(),
    UpSampling2D(), #16x16
    Convolution2D(64, 3, 3, border_mode='same'),
    BatchNormalization(),
    ELU(),
    UpSampling2D(), # 32x32
    Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid')
], name="generator")

print("setup dcgan")
set_trainable(discriminator, False)
dcgan = Sequential([generator, discriminator])
opt_g = Adam(lr=0.001)
dcgan.compile(optimizer=opt_g, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("setup discriminator")
set_trainable(discriminator, True)
opt_d = SGD(lr=0.01)
discriminator.compile(optimizer=opt_d, 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])


imageGen = ImageDataGenerator(width_shift_range=0.05, 
                              height_shift_range=0.05,
                              zoom_range=[0.95, 1.0])
abyss_letters = np.load(npy_path)

def evaluate():
    generate_batch_num = 1000
    random_features = create_random_features(generate_batch_num)
    pred = dcgan.predict(random_features)
    faked = np.sum(pred>0.5)
    print("fake:", faked, "/", generate_batch_num)
    print("")

for i in range(100000):

    # tran discriminator
    print("discriminator:",i)
    generate_batch_num = len(abyss_letters)
    random_features = create_random_features(generate_batch_num)
    generated = generator.predict(random_features)

    X_train = np.append(abyss_letters, generated).reshape([-1, 32, 32, 1])
    y_train = np.append(np.ones(len(abyss_letters)), np.zeros(len(generated)))
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]
    
    random_features = create_random_features(generate_batch_num*10)
    X_val = generator.predict(random_features)
    y_val = np.zeros(len(X_val))

    discriminator.fit_generator(imageGen.flow(X_train, y_train),
                                samples_per_epoch=len(X_train),
                                nb_epoch=300,
                                validation_data=(X_val, y_val),
                                callbacks=[
                                    EarlyStopping(patience=3, monitor='val_loss')
                                ])
    
    evaluate()

    # train generator
    print("generator:",i)
    generate_batch_num = 30000
    X_train = create_random_features(generate_batch_num)
    y_train = np.ones(len(X_train))

    dcgan.fit(X_train, y_train, 
              nb_epoch=30, 
              batch_size=128,
              callbacks=[
                  GeneratorEarlyStopping(generate_batch_num//5)
              ])

    evaluate()

    # save model
    if i%30 == 0:
        model = Sequential([generator, discriminator])
        model.save(os.path.join(files_dir, "./models/{0}.h5".format(i)))

dcgan.compile(optimizer="adam", loss="binary_crossentropy")
dcgan.save(model_path)