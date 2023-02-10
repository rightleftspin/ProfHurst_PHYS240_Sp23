#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Problem 6
import numpy as np

#defining the conversion from cart. to polar. 
def convert(x,y):
    r = np.sqrt(x**2 + y**2) 
    #I used arctan2 instead of arctan because it allows us to take 2 input arrays instead of one. 
    theta = np.degrees(np.arctan2(y,x)) 
    return (r,theta)

#allows us to input a number 
x = float(input('Enter x-coordinate: '))
y = float(input('Enter y-coordinate: '))
polar = convert(x,y)
#using modulo operator to convert a float to a string. %.2f allows us to set the decimal place. The values from polar[0] & polar[1] replaces %.2f. 
print('Polar coordinates: (r=%.2f, Theta =%.2f)' % (polar[0], polar[1]))


# In[10]:


#Problem 3
import time
import numpy as np
#time.time() returns the time as a float number and is expressed in seconds
start = time.time()
operations = 1

#A new time.time() is placed to get the difference from the start of the loop to its end. 
while (time.time() - start) <= 1:
    #The operation is being added to the value and producing a new value. 'operations = operations + 7.0' is in a loop until time is 1 second.
    operations += 7.0

print(str(time.time() - start) + ' seconds')
print("Estimated floats: ", operations )

