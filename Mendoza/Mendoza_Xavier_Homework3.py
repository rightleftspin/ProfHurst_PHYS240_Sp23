#!/usr/bin/env python
# coding: utf-8

# In[8]: Problem 3


# Balle - Program to compute the trajectory of a baseball using the Euler method.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt

# Set initial position and velocity of the baseball
y0 = eval(input('Enter initial ball height (meters): '))
speed = eval(input('Enter initial ball speed (m/s): '))

# Set physical parameters (mass, Cd, etc.)
Cd = 0.35  # Drag coefficient (dimensionless)
area = 4.3e-3  # Cross-sectional area of projectile (m^2)
mass = 0.145   # Mass of projectile (kg)
grav = 9.81    # Gravitational acceleration (m/s^2)

# Set air resistance flag
airFlag = eval(input('Add air resistance? (Yes: 1 No: 0)'))
if airFlag == 0:
    rho = 0.       # No air resistance
    air_text = '(no air)'
else:
    rho = 1.2     # Density of air (kg/m^3)
    air_text = '(with air)'
air_const = -0.5*Cd*rho*area/mass   # Air resistance constant

# Loop over launch angles
angles = range(10, 51)
ranges = []
for theta in angles:
    r0 = np.array([0., y0])  # Initial vector position
    v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
    r = np.copy(r0)  # Set initial position
    v = np.copy(v0)  # Set initial velocity

    # * Loop until ball hits ground or max steps completed
    tau = 0.01  # (sec)
    maxstep = 1000
    laststep = maxstep

    for istep in range(maxstep):
        t = istep * tau  # Current time

        # Calculate the acceleration of the ball
        accel = air_const * np.linalg.norm(v) * v  # Air resistance
        accel[1] = accel[1] - grav # update y acceleration to include gravity

        # Calculate the new position and velocity using Euler's method.
        r = r + tau * v  # Euler step
        v = v + tau * accel

        # If the ball reaches the ground (i.e. y < 0), break out of the loop
        if r[1] < 0:
            laststep = istep + 1
            break  # Break out of the for loop

    # Record the maximum range for this angle
    ranges.append(r[0])

# Find the angle at which the maximum range is achieved
max_range = max(ranges)
max_angle = angles[ranges.index(max_range)]

# Print maximum range and angle of maximum range
print('Maximum range is {0:.2f} meters at angle {1} degrees'.format(max_range, max_angle))

# Plot the range vs angle
plt.plot(angles, ranges)
plt.xlabel('Launch Angle (degrees)')
plt.ylabel('Range (meters)')
plt.title('Maximum Range vs Launch Angle')
plt.show()


# In[27]: Problem 4


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def f(x,y):
    return y**2+1


# Initial
x0 = 0
y0 = 0
x_range=[0,1]


# Solve using Euler's method with different step sizes
dx_values = [0.05, 0.1, 0.2]        

for dx in dx_values:
    # Set the number of steps
    n = int(1/dx)
    
    # analytical solution
    x = np.zeros(n+1)
    y = np.zeros(n+1)
    x[0] = x0
    y[0] = y0
    
    for i in range(n):
        y[i+1] = y[i] + dx*f(x[i], y[i])
        x[i+1] = x[i] + dx
        
    # Plot the results
    plt.plot(x, y, label=f"Step size: {dx}")

#  solve_ivp to approximate the solution to this initial value problem over the interval [0,1]
sol = solve_ivp(f, x_range, [y0], dense_output=True)
x_exact = np.linspace(0, 1, 101)
y_exact = sol.sol(x_exact)[0]




# Plot the exact solution
plt.plot(x_exact, y_exact, label="Exact solution")

# Add plot labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Show the plot
plt.show()

