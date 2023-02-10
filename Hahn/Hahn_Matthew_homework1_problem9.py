# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:19:33 2023

@author: Matthew
"""

#defining the first Catalan number as a variable
c0 = 1

#defining the Catalan function
def catalan(n):
    c1 = ((4 * n + 2) / (n + 2)) * c0
    return c1

#defining variables, n is the second number, c is the variable for the function
n = 1
c = catalan(n)
print(c0)
print(catalan(n))

#using the while loop to continuously change the n and c variables to get higher numbers
#I set the limit less than or equal to 400 million because the next number in the series after would exceed the 1 billion limit
while (c <= 400000000):
    n += 1
    c0 = c
    c = catalan(n)
    print(c)