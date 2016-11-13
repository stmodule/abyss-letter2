#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

files_dir = os.path.dirname(__file__)
npy_path = os.path.join(files_dir, "./dcgan_train_data.npy")
images = np.load(npy_path)

np.random.shuffle(images)

rows = 7
cols = 10
plt.gray()
fig = plt.figure()
for i in range(rows):
    for j in range(cols):
        ax = fig.add_subplot(rows,cols,i*cols+j+1)
        ax.axis('off')
        ax.imshow(images[i*cols+j].reshape([32, 32]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()
