import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from nm4p.rk4 import rk4
from nm4p.rka import rka


# Define gravrk function used by the Runge-Kutta routines
def gravrk(s, t, gkml_array):
    """
    Returns the right-hand side of the Kepler ODE; used by Runge-Kutta routines
    :param s: State vector [r(0), r(1), v(0), v(1)]
    :param t: Time (not used here, included to match derivsRK input)
    :param GM: Parameter G*M - gravitational constant * solar mass Units [AU^3/yr^2]
    :return: deriv: Derivatives [dr(0/dt), dr(1)/dt, dv(0)/dt, dv(1)/dt]
    """
    g, k, m, l = gkml_array

    # Compute acceleration
    r = s[:2]  # Unravel the vector s into position and velocity
    v = s[2:]
    accel = [((r[0] + l) * (v[1] ** 2)) - ((k * r[0]) / m) + (g * np.cos(r[1])), 
             ((-g * np.sin(r[1])) / (l + r[0])) - ((2 * v[0] * v[1]) / (l + r[0]))]

    # Return derivatives
    deriv = np.array([v[0], v[1], accel[0], accel[1]])

    return deriv

mass = 1  # Mass of pendulum bob
grav = 9.81
spring = 1000 # Strength of the spring
length = 1 # Resting length of the pendulum
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

theta0 = eval(sys.argv[1])
total_time = eval(sys.argv[2]) # in sec
tau = eval(sys.argv[3])

nStep = int(total_time // tau)

r = np.array([0, theta0 * np.pi / 180])
v = np.array([0.0, 0.0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)

for iStep in tqdm(range(nStep)):

    # Record position and energy for plotting
    rplot[iStep] = r[0]  # Record radial position and angle for polar plot
    thplot[iStep] = r[1]
    tplot[iStep] = time

    # Calculate new position and velocity using the adaptive Runga-Kutta
    #state = rk4(state, time, tau, gravrk, [grav, spring, mass, length])
    state, time, tau = rka(state, time, tau, adaptErr, gravrk, [grav, spring, mass, length])
    r = state[:2]  # 4th Order Runge-Kutta
    v = state[2:]

plt.plot(tplot, rplot, label = "Radius")
plt.plot(tplot, thplot, label = "Angle")
plt.ylabel('Angle (rad) and Radius (m)')
plt.xlabel('Time')
plt.legend()

plt.savefig("tt_plot.pdf")
plt.close()

rplot = rplot + length

plt.plot(rplot * np.sin(thplot), -rplot * np.cos(thplot))
plt.xlabel('X Distance (m)')
plt.ylabel('Y Distance (m)')

plt.savefig("motion_plot.pdf")

