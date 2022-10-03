#!/usr/bin/env python3

import numpy as np

x = np.random.normal(size=2000)
y = np.cumsum(x)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y)
plt.show()