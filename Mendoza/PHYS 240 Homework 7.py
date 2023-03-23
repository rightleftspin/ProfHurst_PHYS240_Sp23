#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

data = np.loadtxt('curve_data.txt', skiprows=1)
x = data[:, 0]
y = data[:, 1]
dy = data[:, 2]

#def aline(x, a, b):
"""
    Function returns a line with independent variable x
    :param x: independent variable
    :param a: slope
    :param b: intercept
    :return: a*x+b
    """
#   return a*x+b

def poly(x, **params):
    """
    Polynomial fitting function that takes in a parameter dictionatry which are the polynomial coefficients
    :param x: independent variable
    :param params: keywork argument - dictionary of the form ({'c00': c0, 'c01': c1, etc})
    :return: polynomial function of arbitrary order
    """
    temp = 0.0
    parnames = sorted(params.keys())
    for i, pname in enumerate(parnames):
        temp += params[pname]*x**i
    return temp

if __name__ == '__main__':
    

    #* Loop over different polynomial degrees and calculate the reduced chi-squared value
    max_degree =   len(x) - 1 # Maximum polynomial degree to test. Avoid overfitting
    reduced_chi2 = []

    for degree in range(1, max_degree + 1):
        model = Model(poly)
        params = Parameters()
        for j in range(degree):
            params.add('C{0:g}'.format(j), value=1)
        result = model.fit(y, params, x=x )

        chi2 = result.chisqr
        n_varys = len(x) - degree
        reduced_chi2.append(chi2 / n_varys)

    #find the best-fit polynomial degree by minimizing the reduced chi-squared value
    best_degree = np.argmin(reduced_chi2) + 1
    print(f"Best polynomial degree: {best_degree}")

    #* Fit the data using the best-fit polynomial degree
    model = Model(poly)
    params = Parameters()
    for j in range(best_degree):
        params.add('C{0:g}'.format(j), value=1)
    best_fit_result = model.fit(y, params, x=x )

    #* Print the results and plot the data with the best-fit model curve
    best_fit_result.params.pretty_print()
    print(best_fit_result.fit_report())

    plt.errorbar(x, y, yerr=dy, fmt='o', capsize=3, label='Data')
    plt.plot(x, best_fit_result.best_fit, label=f'Best-fit polynomial (degree {best_degree})')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Data with Best-fit Polynomial')
    plt.legend()
    plt.show()


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

data = np.loadtxt('curve_data.txt', skiprows=1)
x = data[:, 0]
y = data[:, 1]
dy = data[:, 2]

def aline(x, a, b):
    """
    Function returns a line with independent variable x
    :param x: independent variable
    :param a: slope
    :param b: intercept
    :return: a*x+b
    """
    return a * x + b

# Transform x values by taking the sine
x_sin = np.sin(x)

# Fit the transformed data using a linear model
model = Model(aline, independent_vars=['x'])
result = model.fit(y, x=x_sin, a=1, b=1,)

# Print the fit report
print(result.fit_report())

# Plot the original data with error bars
plt.errorbar(x, y, yerr=dy, fmt='o', capsize=3, label='Data')

# Plot the best-fit sine curve
x_smooth = np.linspace(x.min(), x.max(), 500)
y_smooth = result.params['a'] * np.sin(x_smooth) + result.params['b']
plt.plot(x_smooth, y_smooth, label='Best-fit sine curve')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Data with Best-fit Sine Curve')
plt.legend()

# Show the plot
plt.show()


# In[39]:


#Question 2, part a

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

days = np.array([1, 2, 3, 4, 5])
dja = np.array([2470, 2510, 2410, 2350, 2240])

def poly(x, **params):
    temp = 0.0
    parnames = sorted(params.keys())
    for i, pname in enumerate(parnames):
        temp += params[pname]*days**i
    return temp

for degree in range(1, 5):
    model = Model(poly)
    params = Parameters()
    for j in range(degree + 1):
        params.add(f'C{j}', value=1)
    result = model.fit(dja, params, x=days)

    print(f"Fit report for polynomial of degree {degree}:")
    print(result.fit_report())

    plt.plot(days, result.best_fit, label=f'Degree {degree} polynomial')

plt.scatter(days, dja, color='red', label='Data')
plt.xlabel('Day')
plt.ylabel('DJA')
plt.title('Dow Jones Averages fitted with Polynomials (Degree 1 to 4)')
plt.legend()
plt.show()


# In[46]:


#part b

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

x = np.array([1, 2, 3, 4, 5])
y = np.array([2470, 2510, 2410, 2350, 2240])
x_extended = np.array([1, 2, 3, 4, 5, 6])

def poly(x, **params):
    temp = 0.0
    parnames = sorted(params.keys())
    for i, pname in enumerate(parnames):
        temp += params[pname]*x**i
    return temp

for degree in range(1, 5):
    model = Model(poly)
    params = Parameters()
    for j in range(degree + 1):
        params.add(f'C{j}', value=1)
    result = model.fit(y, params, x=x)

    print(f"Fit report for polynomial of degree {degree}:")
    print(result.fit_report())

    #computes the best-fit polynomial values for the extended range
    best_fit_extended = poly(x_extended, **result.params)
    plt.plot(x_extended, best_fit_extended, label=f'Degree {degree} polynomial')

plt.scatter(days, dja, color='red', label='Data')
plt.xlabel('Day')
plt.ylabel('DJA')
plt.title('Dow Jones Averages fitted with Polynomials (Degree 1 to 4) with 6th Day Prediction')
plt.legend()
plt.show()


# In[64]:


import numpy as np
import matplotlib.pyplot as plt

def equilibrium_positions(k1, k2):
    A = np.array([[k1 + k2, -k1, 0, 0],
                  [-k1, 2*k1 + k2, -k1, 0],
                  [0, -k1, 2*k1 + k2, -k1],
                  [0, 0, -k1, k1 + k2]])


# In[ ]:




