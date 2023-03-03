# Balle - Program to compute the trajectory of a baseball using the Euler method.

# Set up configuration options and special features
import numpy as np
import math
import matplotlib.pyplot as plt

# Set up the initial position and velocity of the iron balls
y0 = eval(input('Enter initial ball heights (meters): '))
y0_2 = y0

# Set up postion arrays
r0 = np.array([0., y0])
r0_2 = np.array([0., y0_2])

# Set initial speeds of iron balls
speed = eval(input('Enter initial ball speeds (m/s): '))
speed_2 = speed

# Set up intial theta's
theta = eval(input('Enter initial angles (degrees): '))
theta_2 = theta

v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
v0_2 = np.array([speed_2 * np.cos(theta_2*np.pi/180), speed_2 * np.sin(theta_2*np.pi/180)])  # initial velocity of the second ball
r = np.copy(r0)  # Set initial position
r2 = np.copy(r0_2)  # Set initial position of second ball
v = np.copy(v0)  # Set initial velocity
v_2 = np.copy(v0_2)  # Set initial velocity of second ball

# Set physical parameters (mass, Cd, etc.)
Cd = 0.5  # Drag coefficient (dimensionless)
area = 1.38e-2  # Cross-sectional area of projectile (0.4535 kg)(m^2)
area_2 = 8.72e-2 # Cross-sectional area of projectile (45.3592 kg)(m^2)
mass = 0.4535   # Mass of projectile (kg)
mass_2 = 45.3592 # Mass of second projectile (kg)
grav = 9.81    # Gravitational acceleration (m/s^2)

# Set air resistance flag
airFlag = eval(input('Add air resistance? (Yes: 1 No: 0)'))
if airFlag == 0:
    rho = 0.       # No air resistance
    air_text = '(no air)'
else:
    rho = 1.2     # Density of air (kg/m^3)
    air_text = '(with air)'
air_const_1 = -0.5*Cd*rho*area/mass   # Air resistance constant
air_const_2 = -0.5*Cd*rho*area_2/mass_2   # Air resistance constant of second ball

# * Loop until ball hits ground or max steps completed
tau = eval(input('Enter timestep dt in seconds: '))  # (sec)
maxstep = 1000
laststep = maxstep

# Set up arrays for data
xplot = np.empty(maxstep)
xplot_2 = np.empty(maxstep)
yplot = np.empty(maxstep)
yplot_2 = np.empty(maxstep)

x_noAir = np.empty(maxstep)
x_noAir_2 = np.empty(maxstep)
y_noAir = np.empty(maxstep)
y_noAir_2 = np.empty(maxstep)

for istep in range(maxstep):
    t = istep * tau  # Current time

    # Record computed position for plotting
    xplot[istep] = r[0]
    yplot[istep] = r[1]
    x_noAir[istep] = r0[0] + v0[0] * t
    y_noAir[istep] = r0[1] + v0[1] * t - 0.5 * grav * t ** 2

    # Calculate the acceleration of the ball
    accel = air_const_1 * np.linalg.norm(v) * v  # Air resistance
    accel[1] = accel[1] - grav  # update y acceleration to include gravity

    # Calculate the new position and velocity using Euler's method.
    r = r + tau * v  # Euler step
    v = v + tau * accel

    if r[1] < 0:
        laststep = istep + 1
        xplot[laststep] = r[0]  # Record last values completed
        yplot[laststep] = r[1]
        break

for jstep in range(maxstep):
    t = jstep * tau  # Current time

    xplot_2[jstep] = r2[0]
    yplot_2[jstep] = r2[1]

    x_noAir_2[jstep] = r0_2[0] + v0_2[0] * t
    y_noAir_2[jstep] = r0_2[1] + v0_2[1] * t - 0.5 * grav * t ** 2
    accel_2 = air_const_2 * np.linalg.norm(v_2) * v_2  # Air resistance for ball 2
    accel_2[1] = accel_2[1] - grav  # update y acceleration to include gravity for ball 2

    r2 = r2 + tau * v_2  # Euler step
    v_2 = v_2 + tau * accel_2

    # If the ball reaches the ground (i.e. y < 0), update flags
    if r2[1] < 0:
        laststep_2 = jstep + 1
        xplot_2[laststep_2] = r2[0]  # Record last values completed
        yplot_2[laststep_2] = r2[1]
        break

# Calculate seperation distance between ball 1 and 2 using two seperate methods
if rho != 0:
    distance = yplot[laststep_2]
    b = Cd * rho * area / (2 * mass)
    yt = y0 - (1 / b) * math.log(math.cosh(math.sqrt(b * grav) * (laststep_2 * tau)))  # part (b) method

# Print out the maximum range, time of flight, and difference in postion of both balls
print('Maximum range of ball 1 is {0:.5f} meters'.format(r[0]))
print('Maximum range of ball 2 is {0:.5f} meters'.format(r2[0]))
print('\nTime of flight for ball 1 is {0:.5f} seconds'.format(laststep * tau))
print('Time of flight for ball 2 is {0:.5f} seconds'.format(laststep_2 * tau))

# If and Else statement to determine what print statements to run dependent on user input for rho
if rho != 0:
    print('\nThe estimated distance betweeen the two balls when ball 2 hits is: \n{0:.5f} meters'.format(distance))
    print('The actual distance betweeen the two balls when ball 2 hits is: \n{0:.5f} meters'.format(yt))
else:
    print('\nThe balls drop at the same time with no air resistance')

# Graph the trajectory of the baseballs
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
ax1.set_title('Projectile Motion (1-lb): ' + air_text)
ax2.set_title('Projectile Motion (100-lb): ' + air_text)
ax3.set_title('Projectile Motion of both balls: ' + air_text)
ax1.plot(x_noAir[:laststep], y_noAir[:laststep], '-', c='red', label='Theory (no air)')
ax2.plot(x_noAir_2[:laststep_2], y_noAir_2[:laststep_2], '-', c='red', label='Theory (no air)')
ax1.plot(xplot[:laststep + 1], yplot[:laststep + 1], '+', label='Euler method')
ax2.plot(xplot_2[:laststep_2 + 1], yplot_2[:laststep_2 + 1], '+', label='Euler method')
ax3.plot(x_noAir[:laststep], y_noAir[:laststep], '-', c='red', label='Theory (no air)')
ax3.plot(x_noAir_2[:laststep_2], y_noAir_2[:laststep_2], '-', c='red', label='Theory (no air)')
ax3.plot(xplot[:laststep + 1], yplot[:laststep + 1], '+', label='Euler method')
ax3.plot(xplot_2[:laststep_2 + 1], yplot_2[:laststep_2 + 1], '+', label='Euler method')

# Mark the location of the ground by a straight line
ax1.plot(np.array([0.0, x_noAir[laststep - 1]]), np.array([0.0, 0.0]), '-', color='k')
ax2.plot(np.array([0.0, x_noAir_2[laststep_2 - 1]]), np.array([0.0, 0.0]), '-', color='k')
ax3.plot(np.array([0.0, x_noAir[laststep - 1]]), np.array([0.0, 0.0]), '-', color='k')
ax3.plot(np.array([0.0, x_noAir_2[laststep_2 - 1]]), np.array([0.0, 0.0]), '-', color='k')
ax1.legend(frameon=False)
ax2.legend(frameon=False)
ax3.legend(frameon=False)
ax1.set_xlabel('Range (m)')
ax1.set_ylabel('Height (m)')
ax2.set_xlabel('Range (m)')
ax2.set_ylabel('Height (m)')
ax3.set_xlabel('Range (m)')
ax3.set_ylabel('Height (m)')
plt.savefig('balldrop.png')

# Calculating c to make Galileo's claim to work for part (c)

Cd_new = (12500000 * grav * mass * (laststep * tau)**2)/(16129 * area * rho)
print('Drag constant needed to prove Galileo right: ', Cd_new)