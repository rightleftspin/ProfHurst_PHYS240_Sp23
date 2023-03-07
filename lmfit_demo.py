# lmfit_demo - Script showing how the LMfit package works for curve-fitting

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameter, report_fit

# initialize the data to be fit. Data id quadratic plus some random numbers.
print('Data for curve fit is created using a quadratic function: y(x) = c(0) + c(1)x + c(2)x**2: ')
c = np.array(eval(input('Enter the data coefficients as list [c(0), c(1), c(2)]: ')))
N = 50  # Number of data points
x = np.arange(N)  # x = [0, 1, 2, ... N-1]
y = np.empty(N)
alpha = eval(input('Enter the estimated error bar: '))
sigma = alpha * np.ones(N)  # Constant error bar




