#!/usr/bin/env python

import glob
import os
import numpy as np

files_dir = os.path.dirname(__file__)
out_dir = os.path.join(files_dir, "./ae_train_data")

trains = np.array([], dtype=np.float32)
tests = np.array([], dtype=np.float32)
vals = np.array([], dtype=np.float32)

etls = os.path.join(files_dir, "../ETL2_npy/*.npy")
for f in glob.glob(etls):
    images = np.load(f) # [-1, 30, 30]

    # padding, 32x32
    images = np.lib.pad(images, ((0,0),(1,1),(1,1)), 'constant', constant_values=(0, 0))
    
    np.random.shuffle(images)
    ten = images.shape[0]//10
    tests = np.append(tests, images[:ten])
    vals = np.append(vals, images[ten:3*ten])
    trains = np.append(trains, images[3*ten:])

trains = trains.reshape([-1, 32, 32, 1])
vals = vals.reshape([-1, 32, 32, 1])
tests = tests.reshape([-1, 32, 32, 1])

np.save(os.path.join(out_dir, "./train"), trains)
np.save(os.path.join(out_dir, "./val"), vals)
np.save(os.path.join(out_dir, "./test"), tests)

print("trains:", trains.shape[0])
print("vals:", vals.shape[0])
print("tests:", tests.shape[0])