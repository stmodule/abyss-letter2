#!/usr/bin/env python

import numpy as np
import glob
import os
from PIL import Image, ImageOps

files_dir = os.path.dirname(__file__)
out_path = os.path.join(files_dir, "./dcgan_train_data.npy")
pngs = glob.glob(os.path.join(files_dir, "../abyss_letters/*.png"))

images = np.zeros([0, 32, 32, 1])
for png in pngs:
    image = Image.open(png)
    image = ImageOps.grayscale(image)

    for scale in [1.0, 0.95, 0.9]:
        data = (scale, 0, 0, 0, scale, 0)
        scaled = image.transform(image.size, Image.AFFINE, data)
        zeroone = np.array(scaled).reshape([1, 32, 32, 1])/255
        thresh = np.copy(zeroone)
        thresh[thresh < 0.2] = 0
        thresh[thresh > 0.7] = 1

        arrays = [images, zeroone*2-1]
        for p in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            arrays.append(np.power(thresh, p)*2-1)
        
        images = np.concatenate(arrays)

# shift 1px
nb_al = len(images)
images = np.concatenate([
    images, 
    np.append(images[:,1:,:,:], -np.ones([nb_al,1,32,1]), axis=1),
    np.append(-np.ones([nb_al,1,32,1]), images[:,:-1,:,:], axis=1),
    np.append(images[:,:,1:,:], -np.ones([nb_al,32,1,1]), axis=2),
    np.append(-np.ones([nb_al,32,1,1]), images[:,:,:-1,:], axis=2)
], axis=0)

np.save(out_path, images)
print("Save:", out_path)
print("{0} samples".format(len(images)))