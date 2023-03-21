#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # m/s^2
rho = 1.225  # kg/m^3 #air density
cd = 0.5  # drag coefficient
d = 0.073  # diameter of baseball in meters
A = np.pi * (d/2)**2  # cross-sectional area of baseball
m = 0.145  # mass of baseball in kg
r = d/2  # radius of baseball in meters
S = np.pi * r**2  # surface area of baseball

# Optimal batting parameters from Sawicki and Hubbard paper
bat_speed = float(input("Enter bat speed (m/s): "))
theta_deg = float(input("Enter launch angle (degrees): "))
omega = 1620  # rpm
Cm = 0.074 #coefficient of Magnus force

theta = np.deg2rad(theta_deg)  # Convert launch angle to radians
v0 = bat_speed  # Initial velocity is equal to bat speed

v_spin = r*2*np.pi*omega/60  # tangential velocity due to spin

# Initial conditions
x0 = 0 # initial position 
y0 = 0
vx0 = v0 * np.cos(theta) #initial componenets of baseball's velocity
vy0 = v0 * np.sin(theta)
vx_spin = 0  # initial x-component of spin velocity
vy_spin = v_spin  # initial y-component of spin velocity

# Time array
t = np.linspace(0, 10, 1000)

# Numerical solution of the equations of motion
x = np.zeros_like(t)
y = np.zeros_like(t)
vx = np.zeros_like(t)
vy = np.zeros_like(t)
x[0], y[0], vx[0], vy[0] = x0, y0, vx0, vy0 #sets initial condition 

def compute_forces(vx, vy, vx_spin, vy_spin):
    v = np.sqrt(vx**2 + vy**2) #compute magnitude
    fd = 0.5*rho*v**2*cd*A # drag force
    fdx = -fd*vx/v # components of drag force
    fdy = -fd*vy/v

    spin_dir = np.array([-vy_spin, vx_spin, 0])/v_spin # calculate the spin about an axis
    fm = Cm*rho*S*v_spin**2*spin_dir # calculate magnus force
    fmx = fm[0]
    fmy = fm[1] # calculate components of magnus force

    ax = (fdx + fmx)/m #caluculate components of accerlation
    ay = -g + (fdy + fmy)/m

    return ax, ay

#home run flag and max range variables
home_run = False
max_range = 0
max_range_time = 0

#rk4
for i in range(len(t)-1):
    dt = t[i+1] - t[i]

    #calculate accleration
    ax1, ay1 = compute_forces(vx[i], vy[i], vx_spin, vy_spin)
    vx1, vy1 = vx[i] + 0.5*dt*ax1, vy[i] + 0.5*dt*ay1

    ax2, ay2 = compute_forces(vx1, vy1, vx_spin, vy_spin)
    vx2, vy2 = vx[i] + 0.5*dt*ax2, vy[i] + 0.5*dt*ay2

    ax3, ay3 = compute_forces(vx2, vy2, vx_spin, vy_spin)
    vx3, vy3 = vx[i] + dt*ax3, vy[i] + dt*ay3
    
    ax4, ay4 = compute_forces(vx3, vy3, vx_spin, vy_spin)

    # Update the velocities and positions
    vx[i+1] = vx[i] + (1/6)*(ax1 + 2*ax2 + 2*ax3 + ax4)*dt
    vy[i+1] = vy[i] + (1/6)*(ay1 + 2*ay2 + 2*ay3 + ay4)*dt
    x[i+1] = x[i] + (1/6)*(vx[i] + 2*vx1 + 2*vx2 + vx3)*dt
    y[i+1] = y[i] + (1/6)*(vy[i] + 2*vy1 + 2*vy2 + vy3)*dt
    
    
    # Update the spin velocities
    spin_dir = np.array([-vy_spin, vx_spin, 0])/v_spin
    spin_acc = np.cross(spin_dir, np.array([ax1, ay1, 0]))
    vx_spin += spin_acc[0]*dt
    vy_spin += spin_acc[1]*dt

    
    # Ensure that the x and y axes are positive
    if x[i+1] < 0:
        x[i+1] = 0
        vx[i+1] = 0
        vx_spin = 0
    if y[i+1] < 0:
        max_range = x[i]
        max_range_time = t[i]
        break
     #Update the max range and time
    if y[i+1] <= 0:
        if x[i+1] > max_range:
            max_range = x[i+1]
            max_range_time = t[i+1]
        if vy[i+1] < 0:
            break
    
    # Check if the ball passes 415 feet (126.49 meters) for a home run
if max_range >= 126.49:
    print("Home Run!")
    home_run = True
else:
    print("Not a home run.")   
        
    
#Find the maximum range and the time it takes to reach that range
max_range_feet = max_range * 3.28084  # Convert meters to feet

# Plot the trajectory
plt.plot(x[:i+2], y[:i+2])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# Print the maximum range and the time it takes to reach that range in meters and feet
print(f"The maximum range of the trajectory is {max_range:.2f} meters ({max_range_feet:.2f} feet).")
print(f"The time it takes to reach that range is {max_range_time:.2f} seconds.")


# In[125]:


import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81 # m/s^2
rho = 1.225 # kg/m^3
cd = 0.5 # drag coefficient
d = 0.073 # diameter of baseball in meters
A = np.pi * (d/2)**2 # cross-sectional area of baseball
m = 0.145 # mass of baseball in kg
r = d/2 # radius of baseball in meters
S = np.pi * r**2 # surface area of baseball

# Optimal batting parameters from Sawicki and Hubbard paper
bat_speed = float(input("Enter bat speed (m/s): "))
theta_deg = float(input("Enter launch angle (degrees): "))
omega = 1620 # rpm
Cm = 0.074

theta = np.deg2rad(theta_deg) # Convert launch angle to radians
v0 = bat_speed # Initial velocity is equal to bat speed

v_spin = r*2*np.pi*omega/60 # tangential velocity due to spin

# Initial conditions
x0 = 0
y0 = 0
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)
vx_spin = 0 # initial x-component of spin velocity
vy_spin = v_spin # initial y-component of spin velocity

# Time array
t = np.linspace(0, 10, 1000)

# Numerical solution of the equations of motion using the fourth-order Runge-Kutta method
x = np.zeros_like(t)
y = np.zeros_like(t)
vx = np.zeros_like(t)
vy = np.zeros_like(t)
x[0], y[0], vx[0], vy[0] = x0, y0, vx0, vy0

def compute_forces(vx, vy, vx_spin, vy_spin):
    v = np.sqrt(vx**2 + vy**2)
    fd = 0.5*rho*v**2*cd*A
    fdx = -fd*vx/v
    fdy = -fd*vy/v

    spin_dir = np.array([-vy_spin, vx_spin, 0])/v_spin
    fm = Cm*rho*S*v_spin**2*spin_dir
    fmx = fm[0]
    fmy = fm[1]

    ax = (fdx + fmx)/m
    ay = -g + (fdy + fmy)/m

    return ax, ay

for i in range(len(t)-1):
    dt = t[i+1] - t[i]

    ax1, ay1 = compute_forces(vx[i], vy[i], vx_spin, vy_spin)
    vx1, vy1 = vx[i] + 0.5*dt*ax1, vy[i] + 0.5*dt*ay1

    ax2, ay2 = compute_forces(vx1, vy1, vx_spin, vy_spin)
    vx2, vy2 = vx[i] + 0.5*dt*ax2, vy[i] + 0.5*dt*ay2
    
    ax3, ay3 = compute_forces(vx2, vy2, vx_spin, vy_spin)
    vx3, vy3 = vx[i] + dt*ax3, vy[i] + dt*ay3

    ax4, ay4 = compute_forces(vx3, vy3, vx_spin, vy_spin)

    # Update the velocities and positions using the fourth-order Runge-Kutta method
    vx[i+1] = vx[i] + (1/6)*(ax1 + 2*ax2 + 2*ax3 + ax4)*dt
    vy[i+1] = vy[i] + (1/6)*(ay1 + 2*ay2 + 2*ay3 + ay4)*dt
    x[i+1] = x[i] + (1/6)*(vx[i] + 2*vx1 + 2*vx2 + vx3)*dt
    y[i+1] = y[i] + (1/6)*(vy[i] + 2*vy1 + 2*vy2 + vy3)*dt

    if y[i+1] <= 0:
        break

# Truncate arrays to the point where the ball hits the ground
x = x[:i+1]
y = y[:i+1]

# Find the maximum range and the time it takes to reach that range
max_range = x[-1]
max_range_time = t[i]

# Print the maximum range and the time it takes to reach that range
print(f"The maximum range of the trajectory is {max_range:.2f} meters.")
print(f"The time it takes to reach that range is {max_range_time:.2f} seconds.")

# Plot the trajectory of the baseball
plt.plot(x, y)
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Baseball Trajectory')
plt.show()


# In[ ]:




