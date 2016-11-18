#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
import os, sys

if len(sys.argv)==2:
    length = int(sys.argv[1])
else:
    length = 0

files_dir = os.path.dirname(__file__)
met_curve = np.load(os.path.join(files_dir, "./met_curve.npy"))

fig = plt.figure()

# loss
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("loss")
ax1.plot(met_curve[-length:, 0], 'r', label="d")
ax1.plot(met_curve[-length:, 2], 'b', label="g")
# ax1.set_ylim([0,2])
ax1.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05), ncol=2)

# acc
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("acc")
ax2.plot(met_curve[-length:, 1], 'r', label="d")
ax2.plot(met_curve[-length:, 3], 'b', label="g")
ax2.set_ylim([0,1])
ax2.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05), ncol=2)
plt.show()