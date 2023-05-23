# Balle - Program to compute the trajectory of a baseball using the Euler method.

# Set up configuration options and special features
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set initial position and velocity of the baseball
#y0 = eval(input('Enter initial ball height (meters): '))
y0 = 0
r0 = np.array([0., y0])  # Initial vector position
speed = eval(input('Enter initial ball speed (m/s): '))
#theta = eval(input('Enter initial angle (degrees): '))
theta = 45

v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
r = np.copy(r0)  # Set initial position
v = np.copy(v0)  # Set initial velocity

# Set physical parameters (mass, Cd, etc.)
Cd = 0.35  # Drag coefficient (dimensionless)
area = 4.3e-3  # Cross-sectional area of projectile (m^2)
mass = 0.145   # Mass of projectile (kg)
grav = 9.81    # Gravitational acceleration (m/s^2)

# Set air resistance flag
#airFlag = eval(input('Add air resistance? (Yes: 1 No: 0)'))
airFlag = 1
if airFlag == 0:
    rho = 0.       # No air resistance
    air_text = '(no air)'
else:
    rho = 1.2     # Density of air (kg/m^3)
    air_text = '(with air)'
air_const = -0.5*Cd*rho*area/mass   # Air resistance constant

# * Loop until ball hits ground or max steps completed
#tau = eval(input('Enter timestep dt in seconds: '))  # (sec)
tau = .1
maxstep = 1000000
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
    r = r + tau * v  # Euler step
    v = v + tau * accel

    # If the ball reaches the ground (i.e. y < 0), break out of the loop
    if r[1] < 0:
        laststep = istep + 1
        xplot[laststep] = r[0]  # Record last values completed
        yplot[laststep] = r[1]

        # x_noAir[laststep] = r0[0] + v0[0] * t
        # y_noAir[laststep] = r0[1] + v0[1] * t - 0.5 * grav * t ** 2
        break  # Break out of the for loop

# Print maximum range and time of flight
print('Maximum range is {0:.2f} meters'.format(r[0]))
print('Time of flight is {0:.1f} seconds'.format(laststep * tau))

xy_data = pd.DataFrame(np.array([xplot[:laststep + 1].transpose(), yplot[:laststep + 1].transpose()]))
xy_data.to_pickle(f"data_{speed}v.pickle")

xy_data_no_air = pd.DataFrame(np.array([x_noAir[:laststep].transpose(), y_noAir[:laststep].transpose()]))
xy_data_no_air.to_pickle(f"data_{speed}v_na.pickle")

# Graph the trajectory of the baseball
fig, ax = plt.subplots()
ax.set_title('Projectile Motion: ' + air_text)
ax.plot(x_noAir[:laststep], y_noAir[:laststep], '-', c='C2', label='Theory (no air)')
ax.plot(xplot[:laststep+1], yplot[:laststep+1], '+', label='Euler method')
# Mark the location of the ground by a straight line
ax.plot(np.array([0.0, x_noAir[laststep-1]]), np.array([0.0, 0.0]), '-', color='k')
ax.legend(frameon=False)
ax.set_xlabel('Range (m)')
ax.set_ylabel('Height (m)')

plt.savefig("test.pdf")
