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
from optim import *
# Define a two-dimentional objective function
# 3d plot of the test function
# objective function
def objective(x, y):
    return x**2 + y**2

# derivative of objective function
def derivative(x, y):
    return asarray([x * 2.0, y * 2.0])


# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 50
# define the step size
rho = 0.99
beta1 = 0.9
beta2 = 0.99
alpha = 0.02
step_size = 0.1
AdaGrad_Results = np.asarray(adagrad(objective, derivative, bounds, 50, step_size))
RMSProp_Results = np.asarray(rmsprop(objective, derivative, bounds, 50, step_size, rho))
Adam_Results = np.asarray(Adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8))
# function that draws each frame of the animation
def animate(i):
    # # ax.clear()
    # # define range for input
    # r_min, r_max = -1.0, 1.0
    # sample input range uniformly at 0.1 increments
    # ax.plot(RMSProp_Results[:i+1,0], RMSProp_Results[:i+1,1], 'r', linewidth=3)
    ax.scatter(RMSProp_Results[:i+1,0], RMSProp_Results[:i+1,1], c='r', s=16)
    # ax.plot(AdaGrad_Results[:i+1,0], AdaGrad_Results[:i+1,1], 'k', linewidth=2)
    ax.scatter(AdaGrad_Results[:i+1,0], AdaGrad_Results[:i+1,1], c='k', s=16)
    # ax.plot(Adam_Results[:i+1,0], Adam_Results[:i+1,1], 'white', linewidth=2)
    ax.scatter(Adam_Results[:i+1,0], Adam_Results[:i+1,1], c='white', s=16)
    # ax.set_xlim([-1,1])
    # ax.set_ylim([-1,1])


from matplotlib.animation import FuncAnimation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.clear()
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
cs = ax.contourf(x, y, results, levels=100, cmap='jet')
cs2 = ax.contour(cs, levels=cs.levels[::15], colors='k', alpha=0.7, linestyles='dashed', linewidths=3)
plt.clabel(cs2, fmt='%2.2f', colors='k', fontsize=16)
# show the plot
# define tick size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tick_params(direction="in",top=True, right=True, which='major')

# set squared figure
plt.axis('square')

# plot titile and x,y label
plt.xlabel(r'$x$', fontsize=16, fontweight='bold')
plt.ylabel(r'$y$', fontsize=16, fontweight='bold')
anim = FuncAnimation(fig, animate, frames=49, interval=200, repeat=False)
anim.save('animated_graph.gif')
# plt.savefig('RMSprop.gif', dpi=300)
# plt.tight_layout()
# plt.show()