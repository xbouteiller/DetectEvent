from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand


fig, ax = plt.subplots()
ax.set_title('click on points', picker=True)
ax.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
line, = ax.plot(rand(100), 'o', picker=5)
a = plt.ginput(4)
plt.close()

print(a)

