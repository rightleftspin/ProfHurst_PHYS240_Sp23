#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import sympy as sp
from sympy import Symbol 
from sympy import symbols 
from sympy import *
x,a = symbols("x,a",real=True)
init_printing(use_unicode=True)


# ### Part 1
# 
# Create a Taylor Series expansion for sine centered around $\pi/2$ and calculate first 3 terms

# In[89]:


def S(x,a,N):
    s=0.0 #Start initial sum as float 0.0
    for i in range(N):
            s += (x-a)**(i)/np.math.factorial(i) * diff(sin(x),x,i).subs(x,a) #Taylor series formula
            #First term is the approximate value around center of expansion x_0 = a
            #Second term differentiates sin(x) i iterations
        
    return s

S(x,sp.pi/2,5)


# In[102]:


round(S(x,sp.pi/2,5).subs(x,0.9).evalf(),5) #Use this to evaluate at sin(0.9) for first three terms


# ### Part 2 
# 
# Code the partial sum for Sine given and find N to match closest value to output for part (1)

# In[101]:


def SineSum(x,N):
    s=0.0 
    for j in range(N):
            #Create summation from formula given for sin(x)
            s += (-1)**(j)/np.math.factorial(2*j+1) * x**(2*j+1) 
        
    return s

round(SineSum(0.9,3),5)


# 0.78345 $\approx$ 0.78342
