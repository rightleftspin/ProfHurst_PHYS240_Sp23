# Balle - Program to compute the trajectory of a baseball using the Euler method.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt

# Set initial position and velocity of the baseball
y0 = eval(input('Enter initial ball height (meters): '))
r0 = np.array([0., y0])  # Initial vector position
speed = eval(input('Enter initial ball speed (m/s): '))
theta = eval(input('Enter initial angle (degrees): '))

v0 = np.array([speed * np.cos(theta*np.pi/180), speed * np.sin(theta*np.pi/180)])  # initial velocity
r = np.copy(r0)  # Set initial position
v = np.copy(v0)  # Set initial velocity

# Set physical parameters (mass, Cd, etc.)
Cd = 0.35  # Drag coefficient (dimensionless)
area = 4.3e-3  # Cross-sectional area of projectile (m^2)
mass = 0.145   # Mass of projectile (kg)
grav = 9.81    # Gravitational acceleration (m/s^2)

# Set air resistance
airResistance = False
if airResistance:
    rho = 0       # No air resistance
    air_text = '(no air)'
else:
    rho = 1.2     # Density of air (kg/m^3)
    air_text = '(with air)'
air_const = -0.5*Cd*rho*area/mass   # Air resistance constant

# * Loop until ball hits ground or max steps completed
tmax, dt = eval(input('Enter time interval and timestep:  tmax, dt (sec): '))  # (sec)
t = np.arange(0, tmax + dt, dt)  # time vector will extend up to tmax
maxstep = len(t)  # maximum number of steps; should be tmax/dt+1

xplot = np.empty(maxstep)
yplot = np.empty(maxstep)

xNoAir = np.empty(maxstep)
yNoAir = np.empty(maxstep)

# analytical solution with no air:
x_analytic = r0[0] + v0[0]*t
y_analytic = r0[1] + v0[1]*t - 0.5*grav*t**2

#######
# Add functionality to include air resistance.
#######

fig, ax = plt.subplots()
ax.plot(x_analytic, y_analytic, '--', c='C2', label='analytic (no air)')

plt.show()
