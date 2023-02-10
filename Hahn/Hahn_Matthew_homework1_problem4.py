# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:09:55 2023

@author: Matthew
"""

import math

# Input the value of the height of the tower
d = float(input("Enter the height of the tower from where the ball will be dropped in meters:"))

# Equation to find the time it takes for the ball to fall
t = float(math.sqrt(((2 * d) / 9.81)))

# Print the time
print('The time for the ball to drop from the height is {0:.2f} seconds'. format(t))