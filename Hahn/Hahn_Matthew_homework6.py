# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:36:52 2023

@author: Matthew
"""

import numpy as np
import matplotlib.pyplot as plt

# Set initial position and velocity of the baseball
y0 = 0.8
r0 = np.array([0., y0])  # Initial vector position
speed = 18
theta = 22

v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
r = np.copy(r0)  # Set initial position for camera data
v = np.copy(v0)  # Set initial velocity for camera data 
r2 = np.copy(r0)  # Set initial position for calculated data
v2 = np.copy(v0)  # Set initial velocity for calculated data

# Set physical parameters (mass, Cd, etc.)
Cd = 0.30  # Drag coefficient (dimensionless)
Cl = -0.29 # Lift coefficient (dimensionless)
area = 0.0375  # Cross-sectional area of projectile (m^2)
mass = 0.424   # Mass of projectile (kg)
grav = 9.81    # Gravitational acceleration (m/s^2)

# Set air resistance flag

rho = 1.2     # Density of air (kg/m^3)
beta = 0.0530
air_const = -0.5*(Cd*rho*area/mass + Cl*rho*area/mass)   # Air resistance constant

# * Loop until ball hits ground or max steps completed
tau = 0.005  # (sec)
runtime = 0.07
maxstep = int(runtime / tau)
laststep = maxstep

# Set up arrays for camera data
xplot = np.empty(maxstep)
zplot = np.empty(maxstep)

for istep in range(maxstep):
    
    t = istep * tau  # Current time

    # Record computed position for plotting
    xplot[istep] = r[0]
    zplot[istep] = r[1]

    # Calculate the acceleration of the ball
    accel = -beta * np.linalg.norm(v) * (Cd * np.cos(theta*np.pi/180) + Cl * np.sin(theta*np.pi/180)) * v  # Air resistance
    accel[1] = accel[1] - grav # update z acceleration to include gravity

    # Calculate the new position and velocity using Euler's method.
        
    r = r + tau * v #Euler method
    v = v + tau * accel
    
 # If the ball reaches the ground (i.e. y < 0), break out of the loop
    if r[1] < 0:
         laststep = istep + 1
         xplot[laststep] = r[0]  # Record last values completed
         zplot[laststep] = r[1]

         break  # Break out of the for loop

# Set up arrays for calculated data
xplot2 = np.empty(maxstep)
zplot2 = np.empty(maxstep)

for istep in range(maxstep):
    
    t = istep * tau  # Current time

    # Record computed position for plotting
    xplot2[istep] = r2[0]
    zplot2[istep] = r2[1]

    # Calculate the acceleration of the ball
    accel2 = beta * np.linalg.norm(v) * (-Cd * np.sin(theta*np.pi/180) + Cl * np.cos(theta*np.pi/180)) * v - grav  # Air resistance
    accel2[1] = accel2[1] - grav # update z acceleration to include gravity

    # Calculate the new position and velocity using Euler's method.
        
    r2 = r2 + tau * v2 #Euler method
    v2 = v2 + tau * accel2
    
 # If the ball reaches the ground (i.e. y < 0), break out of the loop
    if r2[1] < 0:
         laststep = istep + 1
         xplot2[laststep] = r[0]  # Record last values completed
         zplot2[laststep] = r[1]

         break  # Break out of the for loop

fig, ax = plt.subplots()
ax.set_title('Trajectory of a soccer ball, Cd = 0.3, Cl = -0.29 ')
ax.plot(xplot2[:laststep+1], zplot2[:laststep+1], color = 'black', label='Computational data')
ax.plot(xplot[:laststep+1], zplot[:laststep+1],'.', markersize = 10, label='Camera data', color = 'r')
ax.legend(frameon=False)
ax.set_xlabel('Range (m)')
ax.set_ylabel('Height (m)')
print(air_const)



