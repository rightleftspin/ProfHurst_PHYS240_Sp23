# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:51:17 2023

@author: Matthew
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

# Part 1
# 1a

# using loadtxt and finding the path of the curve data file
xdata, ydata, dy = np.loadtxt('C:/Users/Matthew/desktop/myfolder/phys240/curve_data.txt',
                              skiprows = 1, unpack = True)

# here we define the functions that we are going to use to fit the curve
def poly(x, a, b):
    return a*x+b*x**2

def sine(x, a, c):
    return c + a * np.sin(x)

# create the code for the best fit curve using the data points from the text file
model = Model(sine, independent_vars = ['x'])
# I added arbitrary coefficients as I do not think it will change the chi function that much
graph = model.fit(ydata, x=xdata, a = .01, c = .5)

# I know that part a only asks for the error bars but I could not get them to work 
# by just having the points inputted alone so that is why I had to use one of the 
# functions to actually produce the error bars. 

fig, ax = plt.subplots(figsize = (7, 5))
ax.set_title('Curve_data.txt best fit')
ax.plot(xdata, ydata, '.', label='function', markersize = 4)

ax.legend(frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.errorbar(xdata, ydata, xerr = None,  yerr=graph.chisqr, 
             ecolor = 'Black', fmt='', ls = None)
plt.show()




# 1b

# We fit the data with a polynomial best fit curve
model2 = Model(poly, independent_vars = ['x'])
graph2 = model2.fit(ydata, x=xdata, a = .01, b = .5)

fig, ax = plt.subplots(figsize = (7, 5))
ax.set_title('Curve_data.txt best fit')
ax.plot(xdata, ydata, '.', label='function', markersize = 4)
ax.plot(xdata[:len(graph2.best_fit)], graph2.best_fit, '-', label = 'Best fit', markersize = 10)
print(f'poly Chi-squared: {graph2.chisqr}')

ax.legend(frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



# 1c


# This code is the exact same as part a because I had to use a best fit curve
# in order to get the error bars to show up on the graph

xdata, ydata, dy = np.loadtxt('C:/Users/Matthew/desktop/myfolder/phys240/curve_data.txt', 
                              skiprows = 1, unpack = True)

def poly(x, a, b):
    return a*x+b*x**2

def sine(x, a, c):
    return c + a * np.sin(x)


model = Model(sine, independent_vars = ['x'])
graph = model.fit(ydata, x=xdata, a = .01, c = .5)

fig, ax = plt.subplots(figsize = (7, 5))
ax.set_title('Curve_data.txt best fit')
ax.plot(xdata, ydata, '.', label='function', markersize = 4)
print(f'sine Chi-squared: {graph.chisqr}')

ax.legend(frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.errorbar(xdata, ydata, xerr = None,  yerr=graph.chisqr, ecolor = 'Black', fmt='', ls = None)
plt.show()



# 1d

# The preferred model is the sine function as it fits the data much better than the polynomial as the chi-squared value of the sine best fit is much smaller than the polynomial chi-squared



# 2a

# here I created two arrays for the x values and the y values of the Dow Jones
# table that was given to me
days = np.array([1, 2, 3, 4, 5])
DJA = np.array([2470, 2510, 2410, 2350, 2240])

def line(x, a, b):
    return a*x+b

def cube(x, a, b, c):
    return a*x+b*x**2+c*x**3

def quart(x, a, b, c, d):
    return a*x+b*x**2+c*x**3+d*x**4

# I duplicated the method for graphing this new Dow Jones best fit from the earlier
# parts as it is essentially asking the same thing except with different inputs and different curves
stockmodell = Model(line, independent_vars = ['x'])
stockgraphl = stockmodell.fit(DJA, x=days, a = 1, b = 1)

stockmodelp = Model(poly, independent_vars = ['x'])
stockgraphp = stockmodelp.fit(DJA, x=days, a = 1, b = 1)

stockmodelc = Model(cube, independent_vars = ['x'])
stockgraphc = stockmodelc.fit(DJA, x=days, a = 1, b = 1, c = .1)

stockmodelq = Model(quart, independent_vars = ['x'])
stockgraphq = stockmodelq.fit(DJA, x=days, a = 1, b = 1, c = 1, d = 1)

fig, ax = plt.subplots(figsize = (7, 5))
ax.set_title('Dow Jones Averages')
ax.plot(days, DJA, '.', label='Dow Jones', markersize = 4)
ax.plot(days[:len(stockgraphl.best_fit)], stockgraphl.best_fit, '-', 
        label = 'Best fit line', markersize = 10, color = 'black')
ax.plot(days[:len(stockgraphp.best_fit)], stockgraphp.best_fit, '-', 
        label = 'Best fit poly', markersize = 10, color = 'green')
ax.plot(days[:len(stockgraphc.best_fit)], stockgraphc.best_fit, '-', 
        label = 'Best fit cube', markersize = 10, color = 'red')
ax.plot(days[:len(stockgraphq.best_fit)], stockgraphq.best_fit, '-', 
        label = 'Best fit quart', markersize = 10, color = 'orange')

ax.legend(frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# 2b

# I repeated the code of 2a except with adding day 6 and the 500 point drop that 
# is associated with the earnings of day 6
days2 = np.array([1, 2, 3, 4, 5, 6])
DJA2 = np.array([2470, 2510, 2410, 2350, 2240, 1740])


stockmodell = Model(line, independent_vars = ['x'])
stockgraphl = stockmodell.fit(DJA2, x=days2, a = .01, b = .5)

stockmodelp = Model(poly, independent_vars = ['x'])
stockgraphp = stockmodelp.fit(DJA2, x=days2, a = .01, b = .5)

stockmodelc = Model(cube, independent_vars = ['x'])
stockgraphc = stockmodelc.fit(DJA2, x=days2, a = .01, b = .5, c = .5)

stockmodelq = Model(quart, independent_vars = ['x'])
stockgraphq = stockmodelq.fit(DJA2, x=days2, a = .01, b = .5, c = .5, d = .5)

fig, ax = plt.subplots(figsize = (7, 5))
ax.set_title('Dow Jones Averages')
ax.plot(days2, DJA2, '.', label='Dow Jones', markersize = 4)
ax.plot(days2[:len(stockgraphl.best_fit)], stockgraphl.best_fit, '-', 
        label = 'Best fit line', markersize = 10, color = 'black')
ax.plot(days2[:len(stockgraphp.best_fit)], stockgraphp.best_fit, '-', 
        label = 'Best fit poly', markersize = 10, color = 'green')
ax.plot(days2[:len(stockgraphc.best_fit)], stockgraphc.best_fit, '-', 
        label = 'Best fit cube', markersize = 10, color = 'red')
ax.plot(days2[:len(stockgraphq.best_fit)], stockgraphq.best_fit, '-', 
        label = 'Best fit quart', markersize = 10, color = 'orange')

ax.legend(frameon=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


# Part 2
# 6

# Here, I created matrix array containing the equations to finding the currents
# at each resistor in the circuit. I am unsure of what it means by matrix equation
# in the problem so I figured that I would produce a matrix that would do the 
# calculations for me so long as I input the resistances at each resistor
def matrix(iarray):
    itot, r1, r2, r3, r4, r5, r6 = iarray
    i = np.array([itot * ((r2 + r3) / (r1 + r2 + r3)),itot * (r1 / (r1 + r2 + r3)),
                  itot * (r1 / (r1 + r2 + r3)),itot,itot * (r6 / (r5 + r6)),
                  itot * (r5 / (r5 + r6))])
    return i

vtot = 9
rtot = 4325/48
itot = vtot / rtot
current = [itot, 5, 10, 1, 20, 100, 200]
currents = matrix(current)

print('the current of the circuit is: ', list(currents))

