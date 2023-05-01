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
# gradient descent algorithm with adam
def Adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    solutions = list()
    # generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    score = objective(x[0], x[1])
    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent updates
    for t in tqdm(range(n_iter)):
    # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        for i in range(bounds.shape[0]):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
            # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
        # evaluate candidate point
        score = objective(x[0], x[1])
        # keep track of solutions
        solutions.append(x.copy())
        # report progress
        # print('>%d f(%s) = %.5f' % (t, x, score))
        time.sleep(0.01)
    return solutions

def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
    # track all solutions
    solutions = list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # list of the average square gradients for each variable
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for it in tqdm(range(n_iter)):
        # calculate gradient
        gradient = derivative(solution[0], solution[1])
        # update the average of the squared partial derivatives
        for i in range(gradient.shape[0]):
            # calculate the squared gradient
            sg = gradient[i]**2.0
            # update the moving average of the squared gradient
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0-rho))
        # build a solution one variable at a time
        new_solution = list()
        for i in range(solution.shape[0]):
            # calculate the step size for this variable
            alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
            # calculate the new position in this variable
            value = solution[i] - alpha * gradient[i]
            # store this variable
            new_solution.append(value)
        # evaluate candidate point
        solution = asarray(new_solution)
        solutions.append(solution)
        solution_eval = objective(solution[0], solution[1])
        # report progress
        # print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
        time.sleep(0.01)
    return solutions

# gradient descent algorithm with adagrad
def adagrad(objective, derivative, bounds, n_iter, step_size):
    # track all solutions
    solutions = list()
    # generate an initial point
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # list of the sum square gradients for each variable
    sq_grad_sums = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent
    for it in tqdm(range(n_iter)):
        # calculate gradient
        gradient = derivative(solution[0], solution[1])
        # update the sum of the squared partial derivatives
        for i in range(gradient.shape[0]):
            sq_grad_sums[i] += gradient[i]**2.0
        # build solution
        new_solution = list()
        for i in range(solution.shape[0]):
            # calculate the learning rate for this variable
            alpha = step_size / (1e-8 + sqrt(sq_grad_sums[i]))
            # calculate the new position in this variable
            value = solution[i] - alpha * gradient[i]
            new_solution.append(value)
            # store the new solution
        solution = asarray(new_solution)
        solutions.append(solution)
        # evaluate candidate point
        solution_eval = objective(solution[0], solution[1])
        time.sleep(0.01)
        # report progress
        # print('>%d f(%s) = %.5f' % (it, solution, solution_eval))
    return solutions