# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:45:08 2023

@author: Matthew
"""
import math
from math import log10, floor

#setting up a function that rounds the numbers that way the computer does not run out of computing power
def roundfx(x):
    return round(x, -int(floor(log10(abs(x)))))

print("PART C")
a = float(input("Enter the value for a:"))
b = float(input("Enter the value for b:"))
c = float(input("Enter the value for c:"))

# Creating 2 different equations for the quadratic equation
def x():
    x1 = (0-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return x1

def y():
    y1 = (0-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return y1
x = roundfx(x())
y = roundfx(y())

print("The roots for the quadratice equation are:", x, y)


# Creating 2 different equations for the new variation of the quadratic equation
def x():
    x1 = (2 * c) / (0-b - math.sqrt(b**2 - 4 * a * c))
    return x1

def y():
    y1 = (2 * c) / (0-b + math.sqrt(b**2 - 4 * a * c))
    return y1
x2 = roundfx(x())
y2 = roundfx(y())

print("The roots for the quadratice equation are:", x2, y2)

# The two equations do not give the same solution for the quadratic equation of the same values for a, b, and c. 
# I suspect this is because of the amount of bytes that each equations takes up. From the lecture that 
# Dr. Romanowsky taught, he said that there are only a certain amount of bytes that the computer can store when 
# using code. I think that the latter version of the quadratic formula uses more bytes than the former resulting 
# in a different calculation for the values of x.





