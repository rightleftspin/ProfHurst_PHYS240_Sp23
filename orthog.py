# orthog - Program to test if a pair of vectors is orthogonal. Assume vectors are in 3D space.

# Set up configuration options and special features
import numpy as np

# Initialize the vectors a and b
a = np.array(eval(input('Enter the first 3D vector: ')))
b = np.array(eval(input('Enter the second 3D vector: ')))

# Evaluate the dot product as a sum over products of elements.
a_dot_b = 0.
for i in range(3):
    a_dot_b += a[i]*b[i]

# Print dot product and state whether vectors are orthogonal.
if a_dot_b == 0:
    print('Vectors are orthogonal.')
else:
    print('Vectors are NOT orthogonal.')
    print('Dot product = {0:.2f}'.format(a_dot_b))