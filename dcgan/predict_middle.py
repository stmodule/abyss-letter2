#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

with tf.device("/cpu:0"):
    from keras.models import load_model

    def create_random_features(num):
        return np.random.uniform(low=-1, high=1, 
                            size=[num, 4, 4, 4])


    files_dir = os.path.dirname(__file__)
    npy_path = os.path.join(files_dir, "./dcgan_train_data.npy")
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(files_dir, "./dcgan_model.h5")

    dcgan = load_model(model_path)
    dcgan.summary()

    rows = 6
    cols = 7

    features1 = create_random_features(rows)
    features2 = create_random_features(rows)
    
    features = np.hstack([
        features1, 
        0.9 * features1 + 0.1 * features2,
        0.8 * features1 + 0.2 * features2,
        0.7 * features1 + 0.3 * features2,
        0.6 * features1 + 0.4 * features2,
        0.5 * features1 + 0.5 * features2,
        0.4 * features1 + 0.6 * features2,
        0.3 * features1 + 0.7 * features2,
        0.2 * features1 + 0.8 * features2,
        0.1 * features1 + 0.9 * features2,
        features2, 
    ])
    cols = features.shape[1]//4
    features = features.reshape([-1, 4, 4, 4])

    images = dcgan.layers[0].predict(features)

    # print(np.max(images[0] - images[1]))

    plt.gray()
    fig = plt.figure()
    for i in range(rows):
        for j in range(cols):
            if i*cols+j >= len(images):
                break
            ax = fig.add_subplot(rows,cols,i*cols+j+1)
            ax.imshow(images[i*cols+j].reshape([32, 32]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()