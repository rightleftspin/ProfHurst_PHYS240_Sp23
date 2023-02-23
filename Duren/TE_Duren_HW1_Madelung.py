#!/usr/bin/env python
# coding: utf-8

# In[20]:


#Problem 10 - Madelung Constant

import numpy as np
import matplotlib.pyplot as plt
from numpy import *

#Prepare lists for plotting M vs. L, create an empty array for M to be filled in
M_list = []
L_list = [50,25,5,4,3,2,1]

for p in list(L_list):
    L = int(p) #L is distance from origin to furthest atom on the crystal lattice
    
#Create 3 loops for all possible ijk commbinations
#ijk values scale out the distance, a, an atom is from origin

    M = 0.0 #Set Madelung Constant as float to be summed over according to formula
    for i in range(-L,L):
        for j in range(-L, L):
            for k in range(-L, L):
                if i == j == k == 0:
                        continue #restarts iteration at top of the loop with new i,j, or k value
            
            #m is the summation from the original formula
                m = (i**2 + j**2 + k**2)**-0.5 
            
                if (i + j + k)%2 == 0: #Ensure even values by dividing by two, ensure remainder is zero.
                        M += m
                else:
                        M -= m
            #print(i,j,k) #Shows combinations of values
            
    M_list.append(M) #Fills in each Madelung Constant into M_list after ijk for-loop completes
print(M_list)


# In[21]:


# M vs. L plot

plt.plot(L_list,M_list,'g*')
plt.grid(True)
plt.title('M vs. L')
plt.xlabel('Length from Origin in multiples of a')
plt.ylabel('Madelung Constant')
plt.show()

