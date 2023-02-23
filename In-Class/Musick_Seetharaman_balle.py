# Balle - Program to compute the trajectory of a baseball using the Euler method.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from Musick_Seetharaman_interp import intrpf

# Set initial position and velocity of the baseball
y0 = eval(input('Enter initial ball height (meters): '))
r0 = np.array([0., y0])  # Initial vector position
speed = eval(input('Enter initial ball speed (m/s): '))
theta = eval(input('Enter initial angle (degrees): '))

########## Edited Code ############
# List of available methods to choose from, helpful for plotting later
methods = ["Euler", "Euler-Cromer", "Midpoint"]
# ask user for method
method = eval(input('Which Method would you prefer?\nEuler = [0]\nEuler-Cromer = [1]\nMidpoint = [2]\n '))

# if the method is not one of the choices, raise an error
if method not in range(len(methods)):
    raise ValueError(f"Method Choice {method} is not a valid method")
########## Edited Code ############

v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
r = np.copy(r0)  # Set initial position
v = np.copy(v0)  # Set initial velocity

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

# * Loop until ball hits ground or max steps completed
tau = eval(input('Enter timestep dt in seconds: '))  # (sec)
max_runtime = eval(input('Enter Max Runtime t_max in seconds: ')) # (sec)
#maxstep = 1000
maxstep = int(max_runtime // tau)
laststep = maxstep

# Set up arrays for data
xplot = np.empty(maxstep)
yplot = np.empty(maxstep)

x_noAir = np.empty(maxstep)
y_noAir = np.empty(maxstep)

for istep in range(maxstep):
    t = istep * tau  # Current time

    # Record computed position for plotting
    xplot[istep] = r[0]
    yplot[istep] = r[1]

    x_noAir[istep] = r0[0] + v0[0]*t
    y_noAir[istep] = r0[1] + v0[1]*t - 0.5*grav*t**2

    # Calculate the acceleration of the ball
    accel = air_const * np.linalg.norm(v) * v  # Air resistance
    accel[1] = accel[1] - grav # update y acceleration to include gravity

    # Calculate the new position and velocity using Euler's method.
########## Edited Code ############
    if method == 0:
        # apply the euler method, update both r and v simulatneously
        r = r + tau * v  # Euler step
        v = v + tau * accel
    elif method == 1:
        # apply the euler-cromer method, update v first, then r based on v_n+1
        v = v + tau * accel
        r = r + tau * v
    elif method == 2:
        # apply the midpoint method, update v to v_new and then use the average
        # to find the new r
        v_new = v + tau * accel
        r = r + tau * ((v_new + v) / 2)
        v = v_new
########## Edited Code ############

    # If the ball reaches the ground (i.e. y < 0), break out of the loop
    if r[1] < 0:
        laststep = istep + 1
        xplot[laststep] = r[0]  # Record last values completed
        yplot[laststep] = r[1]

        # x_noAir[laststep] = r0[0] + v0[0] * t
        # y_noAir[laststep] = r0[1] + v0[1] * t - 0.5 * grav * t ** 2
        break  # Break out of the for loop

########## Edited Code ############
# Calculate the theoretical max range
max_range_theory = ((2 * speed ** 2) / grav) * np.sin(theta * np.pi / 180) * np.cos(theta * np.pi / 180)
# Calculate the "experimental" max range through the interpf function
max_range = intrpf(0, yplot[laststep - 2:laststep + 1], xplot[laststep - 2: laststep + 1])
# Print maximum range and time of flight

print('Estimated Maximum range is {0:.2f} meters'.format(r[0]))
print('Interpolated Maximum range is {0:.2f} meters'.format(max_range))
print('Theoretical Maximum range is {0:.2f} meters'.format(max_range_theory))
print(f'Percent Error is {np.abs(100 * ((max_range - max_range_theory) / max_range_theory)):.4f}%')
print('Time of flight is {0:.1f} seconds'.format(laststep * tau))
########## Edited Code ############

# Graph the trajectory of the baseball
fig, ax = plt.subplots()
ax.set_title('Projectile Motion: ' + air_text)
ax.plot(x_noAir[:laststep], y_noAir[:laststep], '-', c='C2', label='Theory (no air)')
ax.plot(xplot[:laststep+1], yplot[:laststep+1], '+', label=f'{methods[method]} method')
# Mark the location of the ground by a straight line
ax.plot(np.array([0.0, x_noAir[laststep-1]]), np.array([0.0, 0.0]), '-', color='k')
ax.legend(frameon=False)
ax.set_xlabel('Range (m)')
ax.set_ylabel('Height (m)')

plt.savefig("test.pdf")
