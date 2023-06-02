import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# Set the file path and name
path = 'C:\Users\Youcheng\Desktop\PythonFittingTools\\template_multiple\\'
name = 'Thermal_2T'
append = '.txt'
filename = path + name + append

# Read the data from the file
data = pd.read_csv(filename, delimiter="\t", header=None, dtype=np.float32)
data_array = np.array(data.values)

# Extract frequency values (x)
x = data_array[:, 0]

# Define the functions for the Drude models
def mdl1_func(beta, x):
    mdl1 = (beta[2] * (1 + beta[1]**2 * x**2) * (1 + beta[4]**2 * beta[3]**2 + beta[3]**2 * x**2) +
            beta[0] * (beta[4]**4 * beta[3]**4 + (1 + beta[3]**2 * x**2)**2) +
            beta[4]**2 * (2 * beta[3]**2 - 2 * beta[3]**4 * x**2)) / \
           (1 + beta[1]**2 * x**2) / \
           (beta[4]**4 * beta[3]**4 + (1 + beta[3]**2 * x*x)**2 +
            beta[4]**2 * (2 * beta[3]**2 - 2 * beta[3]**4 * x**2))
    return mdl1

def mdl2_func(beta, x):
    mdl2 = (x * (beta[2] * beta[3] * (1 + beta[1]**2 * x**2) *
                 (1 - beta[4]**2 * beta[3]**2 + beta[3]**2 * x**2) +
                 beta[0] * beta[1] * (beta[4]**4 * beta[3]**4 +
                                      (1 + beta[3]**2 * x**2)**2 +
                                      beta[4]**2 * (2 * beta[3]**2 - 2 * beta[3]**4 * x**2)))) / \
           (1 + beta[1]**2 * x**2) / \
           (beta[3]**4 * beta[4]**4 + (1 + beta[3]**2 * x**2)**2 +
            beta[4]**2 * (2 * beta[3]**2 - 2 * beta[3]**4 * x**2))
    return mdl2

# Define the residual function for the optimization
def residual_two_functions(beta, x, y1, y2):
    mdl1 = mdl1_func(beta, x)
    mdl2 = mdl2_func(beta, x)
    diff1 = y1 - mdl1
    diff2 = y2 - mdl2
    return np.concatenate((diff1, diff2))

# Define the number of temperature points
N = 100

# Initialize arrays to store the best-fit parameters
best = np.zeros((5, N), dtype=float)
best_init = np.array([0.003, 1E-9, 0.002, 2E-10, 1E8])

# Iterate over temperature points and perform fitting
for ind in range(N):
    y1 = data_array[:, 2*ind+1]
    y2 = data_array[:, 2*ind+2]
    best[:, ind], cov, info, message, ier = leastsq(residual_two_functions, best_init, args=(x, y1, y2), full_output=True)
    best_init = best[:, ind]

    # Save fitted data to a file
    save_x = np.linspace(5E7, 8E9, num=1601)
    save_y1 = mdl1_func(best[:, ind], save_x)
    save_y2 = mdl2_func(best[:, ind], save_x)
    data_fit = pd.DataFrame({'Freq_fit': pd.Series(save_x), 'Imag': pd.Series(save_y2), 'Real': pd.Series(save_y1)})
    savepath = path + name + '_' + str(ind) + '_fit' + append
    data_fit.to_csv(savepath, sep='\t', mode='w', index=False, columns=['Freq_fit', 'Real', 'Imag'])

# Save best values to a file
par = pd.DataFrame(best).T
savepath = path + 'fit_parameters' + '_2T' + append
par.to_csv(savepath, sep='\t', mode='w', index=False, header=False)

# Plot the fitted data
plt.plot(x, y1, linewidth=3, color='k')
plt.plot(x, y2, linewidth=3, color='k', linestyle='dashed')
plt.plot(save_x, save_y1, linewidth=3, color='r')
plt.plot(save_x, save_y2, linewidth=3, color='r', linestyle='dashed')
plt.show()

# Calculate and print the associated temperature
T = (4 - 0.38) / 100 * N + 0.38
print(T)
