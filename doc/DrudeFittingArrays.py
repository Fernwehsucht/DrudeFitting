# -*- coding: utf-8 -*-
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

# Set the path, name, and extension of the file
path = 'C:\\Users\\Youcheng\\Desktop\\IT9_2T_Thermal\\'
name = 'SubCorr_2T_cut'
append = '.txt'
filename = path + name + append

# Read the data from the file
data = pd.read_csv(filename, delimiter="\t", header=None, dtype=np.float64)
data_array = np.array(data.values)

# Extract the frequency from the data
x = data_array[:, 0]
NF = x.shape[0]  # Number of frequency points

# Define the model functions
def mdl1_func(beta, x):
    mdl1 = (beta[2] * (1 + beta[1] ** 2 * x ** 2) * (1 + abs(beta[4]) ** 2 * beta[3] ** 2 + beta[3] ** 2 * x ** 2) + beta[0] * (abs(beta[4]) ** 4 * beta[3] ** 4 + (1 + beta[3] ** 2 * x ** 2) ** 2) + abs(beta[4]) ** 2 * (2 * beta[3] ** 2 - 2 * beta[3] ** 4 * x ** 2)) / (1 + beta[1] ** 2 * x ** 2) / (abs(beta[4]) ** 4 * beta[3] ** 4 + (1 + beta[3] ** 2 * x * x) ** 2 + abs(beta[4]) ** 2 * (2 * beta[3] ** 2 - 2 * beta[3] ** 4 * x ** 2))
    return mdl1

def mdl2_func(beta, x):
    mdl2 = (x * (beta[2] * beta[3] * (1 + beta[1] ** 2 * x ** 2) * (1 - abs(beta[4]) ** 2 * beta[3] ** 2 + beta[3] ** 2 * x ** 2) + beta[0] * beta[1] * (abs(beta[4]) ** 4 * beta[3] ** 4 + (1 + beta[3] ** 2 * x ** 2) ** 2 + abs(beta[4]) ** 2 * (2 * beta[3] ** 2 - 2 * beta[3] ** 4 * x ** 2)))) / (1 + beta[1] ** 2 * x ** 2) / (beta[3] ** 4 * beta[4] ** 4 + (1 + beta[3] ** 2 * x ** 2) ** 2 + abs(beta[4]) ** 2 * (2 * beta[3] ** 2 - 2 * beta[3] ** 4 * x ** 2))
    return mdl2

def residual_two_functions(beta, x, y_input):
    y1 = y_input[:NF]
    nt = y_input.shape[0]
    n = y1.shape[0]
    mdl = np.zeros((nt, 1), dtype=float)
    mdl = mdl.flatten()
    diff = copy.copy(mdl) + 0.0
   


# Define the total number of temperature points
NT = 100

# Define the place where you start (1 means starting with the first point)
start = 1

# Define the number of temperature points you want to go backwards
N = 100

# Define the starting and ending temperature
sT = 0.385822
eT = 3.99249

# Initialize the best values array
best = np.zeros((6, N), dtype=float)
best_init = np.array([0.014420542678880639, 1.687498574842766e-09, 0.002040444628945457, 1.3333575762246272e-10, 10332.07821640472])
bestsave = copy.copy(best)

# Perform fitting for each temperature point
for ind in range(N):
    y1 = data_array[:, 2 * start + 2 * ind - 1]
    y2 = data_array[:, 2 * start + 2 * ind]
    yt = np.concatenate((y1, y2), axis=0)
    res = least_squares(residual_two_functions, best_init, method='lm', ftol=2.23e-16, xtol=2.5e-16, gtol=2.5E-16, args=(x, yt))
    best[1:, ind] = res.x
    best[0, ind] = sT + (start - 1) * (eT - sT) / (NT - 1) + (ind) * (eT - sT) / (NT - 1)
    best_init = best[1:, ind]
    bestsave[1:, ind] = copy.copy(best[1:, ind])
    bestsave[0, ind] = best[0, ind]
    
    # Save fitted data to file
    save_x = np.linspace(5E7, 8E9, num=1601)
    save_y1 = mdl1_func(best[1:, ind], save_x)
    save_y2 = mdl2_func(best[1:, ind], save_x)
    data_fit = pd.DataFrame({'Freq_fit': pd.Series(save_x), 'Imag': pd.Series(save_y2), 'Real': pd.Series(save_y1)})
    savepath = path + name + '_' + str(ind) + '_fit' + append
    data_fit.to_csv(savepath, sep='\t', mode='w', index=False, columns=['Freq_fit', 'Real', 'Imag'])
    
    # Save figures
    fig1 = plt.figure(figsize=(6, 5))
    ax1 = fig1.add_axes([0.2, 0.15, 0.73, 0.75])
    line1, = ax1.plot(x, y1, linewidth=3, color='k')
    ax1.plot(x, y2, linewidth=3, color='k', linestyle='dashed')
    ax1.plot(save_x, save_y1, linewidth=3, color='r')
    ax1.plot(save_x, save_y2, linewidth=3, color='r', linestyle='dashed')
    plt.xlabel(r'$\omega/2\pi$ (Hz)', fontsize=20)
    plt.ylabel(r'$G_{1,2}$ ($\Omega^{-1}/sq$)', fontsize=20)
    ax1.set_xlim([0, 8E9])
    ax1.xaxis.set_tick_params(width=2)
    ax1.yaxis.set_tick_params(width=2)
    plt.tick

# Save best values to a file
par = pd.DataFrame(bestsave)
par = par.T
savepath = path + 'fit_parameters' + '_1T' + append
par.to_csv(savepath, sep='\t', mode='w', index=False, header=False)

# Calculate the final temperature
T = sT + (start - 1) * (eT - sT) / (NT - 1) + (N - 1) * (eT - sT) / (NT - 1)
print(T)
