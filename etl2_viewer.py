#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

if len(sys.argv) != 2:
    files_dir = os.path.dirname(__file__)
    etls = os.path.join(files_dir, "./ETL2_npy/*.npy")

    files = glob.glob(etls)
    random.shuffle(files)

    rows = 7
    cols = 7
    plt.gray()
    fig = plt.figure()
    for i in range(rows):
        for j in range(cols):
            f = files[i*cols+j]
            images = np.load(f)
            ax = fig.add_subplot(rows,cols,i*cols+j+1)
            ax.axis('off')
            filename = os.path.splitext(os.path.basename(f))[0]
            ax.set_title(filename)
            ax.imshow(images[0].reshape([30, 30]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()
else:
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("Not found:", file_path)
        exit(-1)

    images = np.load(file_path)

    rows = 5
    cols = 8
    plt.gray()
    fig = plt.figure()
    for i in range(rows):
        for j in range(cols):
            if i*cols+j >= len(images):
                break
            ax = fig.add_subplot(rows,cols,i*cols+j+1)
            ax.imshow(images[i*cols+j].reshape([30, 30]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()