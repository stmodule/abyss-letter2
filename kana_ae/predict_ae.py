#!/usr/bin/env python

from keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt 

files_dir = os.path.dirname(__file__)
npy_dir = os.path.join(files_dir, "./ae_train_data")
model_path = os.path.join(files_dir, "./ae_model.h5")

ae = load_model(model_path)

tests = np.load(os.path.join(npy_dir, "./test.npy"))
np.random.shuffle(tests)

images = ae.predict(tests)

rows = 3
cols = 8
plt.gray()
fig = plt.figure()
for i in range(rows):
    for j in range(cols):
        if i*cols+j >= len(images):
            break
        ax = fig.add_subplot(rows*2,cols,2*i*cols+j+1)
        ax.imshow(tests[i*cols+j].reshape([32, 32]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = fig.add_subplot(rows*2,cols,(2*i+1)*cols+j+1)
        ax.imshow(images[i*cols+j].reshape([32, 32]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()