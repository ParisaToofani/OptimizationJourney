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

RMSProp_optimal_point_x = RMSProp_Results[:,0]
RMSProp_optimal_point_y = RMSProp_Results[:,1]
RMSProp_results = objective(RMSProp_optimal_point_x, RMSProp_optimal_point_y)

AdaGrad_optimal_point_x = AdaGrad_Results[:,0]
AdaGrad_optimal_point_y = AdaGrad_Results[:,1]
AdaGrad_results = objective(AdaGrad_optimal_point_x, AdaGrad_optimal_point_y)

Adam_optimal_point_x = Adam_Results[:,0]
Adam_optimal_point_y = Adam_Results[:,1]
Adam_results = objective(Adam_optimal_point_x, Adam_optimal_point_y)

print("RMSprop Optimal point is (%f, %f)\n" %(RMSProp_Results[-1,0], RMSProp_Results[-1,1]))
print("RMSprop Optimal value is: %f\n" %RMSProp_results[-1])
print("========================================================================")
print("AdaGrad Optimal point is (%f, %f)\n" %(AdaGrad_Results[-1,0], AdaGrad_Results[-1,1]))
print("AdaGrad Optimal value is: %f\n" %AdaGrad_results[-1])
print("========================================================================")
print("Adam Optimal point is (%f, %f)\n" %(Adam_Results[-1,0], Adam_Results[-1,1]))
print("Adam Optimal value is: %f\n" %Adam_results[-1])
print("========================================================================")

ax = plt.figure(figsize=(6, 6)).add_subplot(projection='3d')

# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x, y) points are plotted on the x and z axes.
ax.scatter(RMSProp_optimal_point_x, RMSProp_optimal_point_y, RMSProp_results, zdir='z', c='r', label='RMSProp')
ax.scatter(AdaGrad_optimal_point_x, AdaGrad_optimal_point_y, AdaGrad_results, zdir='z', c= 'k', label='AdaGrad')
ax.scatter(Adam_optimal_point_x, Adam_optimal_point_y, Adam_results, zdir='z', c = 'b', label='Adam')

# Make legend, set axes limits and labels
ax.legend()
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
# plt.savefig('Optimizers_3d_behavior.png', dpi=600)

# ====================================================================================
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
ax.scatter(RMSProp_Results[:,0], RMSProp_Results[:,1], c='r', s=16, label="RMSprop")
ax.scatter(AdaGrad_Results[:,0], AdaGrad_Results[:,1], c='k', s=16, label="AdaGrad")
ax.scatter(Adam_Results[:,0], Adam_Results[:,1], c='white', s=16, label="Adam")
plt.legend()
# plt.savefig('optimizers_contour.png', dpi=600)