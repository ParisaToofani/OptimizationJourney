# gradient descent optimization with adam for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
# Define a two-dimentional objective function
# 3d plot of the test function
# objective function
def objective(x, y):
    return x**2 + y**2

# derivative of objective function
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])

# ==============================================
# Plot the Surface of the Objective Function
# ==============================================

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# create a surface plot with the jet color scheme
# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=(6, 6))
# First subplot
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_aspect('equal')
ax.plot_surface(x, y, results, cmap='jet')
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
ax.zaxis.set_tick_params(labelsize=14)
ax.xaxis.set_tick_params(which='major', labelsize=14)
ax.yaxis.set_tick_params(which='major', labelsize=14)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 2])
plt.tick_params(direction="in",top=True, right=True)

# set squared figure
plt.axis('square')

# plot titile and x,y label
ax.set_xlabel(r'$x$', fontsize=16, fontweight='bold', labelpad=7)
ax.set_ylabel(r'$y$', fontsize=16, fontweight='bold', labelpad=7)
ax.set_zlabel(r'$x^2+y^2$', fontsize=16, fontweight='bold', labelpad=10)
# plt.tight_layout()
plt.savefig('main_surface.png', dpi=600)


fig1 = plt.figure(figsize=(6, 6))
ax = fig1.add_subplot(1, 1, 1)
cs = ax.contourf(x, y, results, levels=100, cmap='jet')
cs2 = ax.contour(cs, levels=cs.levels[::15], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
plt.clabel(cs2, fmt='%2.2f', colors='k', fontsize=16)
# show the plot
# define tick size

ax.xaxis.set_tick_params(which='major', labelsize=14)
ax.yaxis.set_tick_params(which='major', labelsize=14)
plt.tick_params(direction="in",top=True, right=True)

# set squared figure
plt.axis('square')

# plot titile and x,y label
plt.xlabel(r'$x$', fontsize=16, fontweight='bold')
plt.ylabel(r'$y$', fontsize=16, fontweight='bold')
# plt.tight_layout()
plt.savefig('main_contours.png', dpi=600)

