# lmfit_demo - Script showing how the LMfit package works for curve-fitting

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters


def aline(x, a, b):
    """
    Function returns a line with independent variable x
    :param x: independent variable
    :param a: slope
    :param b: intercept
    :return: a*x+b
    """
    return a*x+b


def poly(x, **params):
    """
    Polynomial fitting function that takes in a parameter dictionatry which are the polynomial coefficients
    :param x: independent variable
    :param params: keywork argument - dictionary of the form ({'c00': c0, 'c01': c1, etc})
    :return: polynomial function of arbitrary order
    """
    temp= 0.0
    parnames = sorted(params.keys())
    for i, pname in enumerate(parnames):
        temp += params[pname]*x**i
    return temp


if __name__ == '__main__':

    # Initialize the data to be fit. This data is a quadratic plus some random numbers.
    print('Data for curve fit is created using a polynomial function of your choice: y(x) = c(0) + c(1)x + c(2)x**2 + ... c(n)**n: ')
    c = np.array(eval(input('Enter the data coefficients as list [c(0), c(1), c(2)... c(n-1)]: ')))
    N = 20  # Number of data points
    x = np.arange(N)  # x = [0, 1, 2, ... N-1]
    y = np.empty(N)
    alpha = eval(input('Enter the estimated error bar: '))
    sigma = alpha * np.ones(N)  # Constant error bar
    randomState = np.random.RandomState()  # Initialize random state

    params_dict = {}
    for el in c:
        params_dict.update({'C{0:g}': el})

    for i in range(N):
        r = alpha * randomState.normal()  # Generate a Gaussian distributed random vector of length N, mean 0 variance alpha
        temp = 0.0
        for m in range(len(c)):
            temp += c[m] * x[i] ** m
        y[i] = temp + r  # Generate data

    #* Fit the data to a straight line or a more general polynomial
    M = eval(input('Enter the number of fit parameters (=2 for line, >2 for polynomial): '))

    if M == 2:
        # linear fit
        model = Model(aline, independent_vars=['x'])
        result = model.fit(y, x=x, a=c[1], b=c[0])

    else:
        # Polynomial fit using poly function
        model = Model(poly)

        # Parameter names and starting values
        params = Parameters()
        for j in range(len(c)):
            params.add('C{0:g}'.format(j), value=c[j])
        result = model.fit(y, params, x=x)

    result.params.pretty_print()
    print(result.fit_report())
    result.plot()
    plt.show()

    # A nice test case is to try c = [0.1, 0.5, 0.25]

