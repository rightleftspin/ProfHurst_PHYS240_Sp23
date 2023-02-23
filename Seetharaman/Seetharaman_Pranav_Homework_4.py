import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    # Compute acceleration
    r = s[:2]  # Unravel the vector s into position and velocity
    v = s[2:]
    accel = [(r[0] + gkml_array[3]) * (v[1] ** 2) + gkml_array[0] * np.cos(r[1]) - (gkml_array[1] / gkml_array[2]) * r[0],
             ((-2 * v[0] * v[1]) / (r[0] + gkml_array[3])) - ((gkml_array[0] / (r[0] + gkml_array[3])) * np.sin(r[1]))]

    # Return derivatives
    deriv = np.array([v[0], v[1], accel[0], accel[1]])

    return deriv

mass = .1  # Mass of pendulum bob
grav = 9.8
spring = 100 # Strength of the spring
length = 1 # Resting length of the pendulum
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Set initial position and velocity of the comet.
theta0 = eval(input('Enter initial angle (deg): '))
r = np.array([length, theta0 * np.pi / 180])
v = np.array([0.0, 0.0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines


# Loop over the desired number of steps using the specified numerical method.
nStep = eval(input('Enter number of iterations: '))
tau = eval(input('Enter time step (sec): '))

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

#rplot = rplot + length

fig = plt.figure(figsize=(10.0, 5.25))
ax = fig.add_subplot(121, polar=True)
ax.plot(thplot, rplot, '+',)
ax.set_title('Distance from Center (m)')
ax.grid(True)
fig.tight_layout(pad=5.0)

ax2 = fig.add_subplot(122)
#ax2.plot(rplot * np.sin(thplot), -rplot * np.cos(thplot))
ax2.plot(tplot, rplot, label = "Rho (m)")
ax2.plot(tplot, thplot, label = "Theta (rad)")
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('')
ax2.legend()

plt.savefig("test.pdf")

