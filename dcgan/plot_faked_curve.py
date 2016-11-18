#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt
import os

files_dir = os.path.dirname(__file__)
print(np.load(os.path.join(files_dir, "./faked_curve.npy")).shape)
faked_curve = np.load(os.path.join(files_dir, "./faked_curve.npy")).reshape([-1,2])

fig = plt.figure()

# loss
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("loss")
ax1.plot(faked_curve[:, 0], 'r')
ax1.set_ylim([0,2])
ax1.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05), ncol=2)

# acc
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("acc")
ax2.plot(faked_curve[:, 1], 'r')
ax2.set_ylim([0,1])
ax2.legend(loc="upper center", bbox_to_anchor=(0.5,-0.05), ncol=2)
plt.show()