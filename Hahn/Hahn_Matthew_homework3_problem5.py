#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import Symbol 
from sympy import symbols 
from sympy import *
x,z= symbols("x,z",real=True)
init_printing(use_unicode=True)

#Type in your function here in terms of y and z
#The function that we are trying to find the second derivative of

y = sin(z)

#* Evaluates 2nd Derivative for any function at any point

dx_list = eval(input('Enter spatial step: '))   #delta_x step
maxstep = 5000    # Maximum number of steps   


sdrvplot = []

for n in list(dx_list):
    print(n)
    dx = float(n)
    
    secdrvs = np.empty(maxstep) #Place these in between loops to reset arrays for every new dx
    xarray = []
    
    for i in range(maxstep):
    
        x = i*dx         # Current distance
        xf = (i+1)*dx    # n+1 from current distance or x+delta_x
        xp = (i-1)*dx    # n-1 from current distance
        xarray.append(x) 
    
        secdrv = (y.subs(z,xf)+y.subs(z,xp)-2*y.subs(z,x))/(dx**2) #substitute distance iterations
        secdrvs[i] = secdrv #Record ith second derivative
    
#print(secplt)
#print(xarray)

#jth index located in same array location for second derivs and x
    k = eval(input('Enter point where you are evaluating the 2nd derivative (in multiples of dx): ') )
    j = xarray.index(round(float(k),2))
    secdrv_eval = secdrvs[j] 
    print('The 2nd derivative is: ',secdrv_eval) #Your second derivative approximation 
    sdrvplot.append(secdrv_eval)


abs_error = []

for m in list(sdrvplot):
    
    errvalue = abs(diff(y,z,2).subs(z,float(k)) - m)
    
    abs_error.append(errvalue)
print('Absolute error: ',abs_error)


plt.loglog(dx_list,abs_error,'g*')
plt.grid(True)
plt.title(r'$\Delta h$ vs dx')
plt.xlabel('dx')
plt.ylabel(r'$\Delta h$')
plt.show()





