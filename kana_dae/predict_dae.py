#!/usr/bin/env python

from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt 

files_dir = os.path.dirname(__file__)
npy_dir = os.path.join(files_dir, "./dae_train_data")
model_path = os.path.join(files_dir, "./dae_model.h5")

ae = load_model(model_path)

cols = 8

def add_noise(images, noise_ratio=0.2):
    noise = np.random.uniform(images.shape)
    noised_images = np.copy(images)
    noised_images[noise < noise_ratio] = 1
    return noised_images

tests = np.load(os.path.join(npy_dir, "./test.npy"))
np.random.shuffle(tests)

tests = tests[:cols]

noised_tests = np.concatenate([
    add_noise(tests, 0.1),
    add_noise(tests, 0.15),
    add_noise(tests, 0.2)
], axis=0)


outputs = ae.predict(noised_tests)

plt.gray()
fig = plt.figure()
for i in range(len(noised_tests)):
    for j in range(cols):
        if i*cols+j >= len(images):
            break
        ax = fig.add_subplot(len(noised_tests)*2, cols, 2*i*cols+j+1)
        ax.imshow(noised_tests[i*cols+j].reshape([32, 32]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = fig.add_subplot(len(noised_tests)*2, cols, (2*i+1)*cols+j+1)
        ax.imshow(outputs[i*cols+j].reshape([32, 32]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()