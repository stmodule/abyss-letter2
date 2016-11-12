#!/usr/bin/env python

import numpy as np
import glob
import os
from PIL import Image, ImageOps

files_dir = os.path.dirname(__file__)
out_path = os.path.join(files_dir, "./dcgan_train_data.npy")
pngs = glob.glob(os.path.join(files_dir, "../abyss_letters/*.png"))

images = np.array([], np.float32)
for png in pngs:
    image = Image.open(png)
    image = ImageOps.grayscale(image)

    array = np.array(image)/255

    images = np.append(images, array)

images = images.reshape([-1, 32, 32, 1])

np.save(out_path, images)
print("Save:", out_path)