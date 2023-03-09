# Original author: Alexi Musick @ Alexi.Musick@sjsu.edu

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nm4p.rk4 import rk4
from nm4p.rka import rka

'''
8. Radial Accuracy (short): For an ellipse, the radial position varies with angle as
r(θ) = a(1 −2)
1 −cos θ (4)
Modify the orbit.py program to compute and plot the absolute fractional error in r(θ)
(using a log scale) versus time.
(a) Using the Euler-Cromer method, obtain results for an initial radial distance of
1 AU, and inital tangential velocity of π AU/yr, and time steps of ∆t = 0.01,0.005,
and 0.001 yr.
(b) Now implement the Verlet method and compare the results for the r(θ) error.
Your final sumbission should have two plots (one from Euler-Cromer, one from
Verlet).
'''


# Import Orbit.py program from Numerical Methods for Physics (Garcia)

# Define gravrk function used by the Runge-Kutta routines
def gravrk(s, t, GM):
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
    accel = -GM * r / np.linalg.norm(r) ** 3  # Gravitational acceleration

    # Return derivatives
    deriv = np.array([v[0], v[1], accel[0], accel[1]])

    return deriv


# Define function to calculate absolute fractional error in r(θ)
def calc_error(r, epsilon, theta):
    '''
    Returns the absolute fractional error
    :param r: radial position
    :param epsilon: epsilon value
    :param theta: theta of orbit
    return: error: calculated error
    '''

    r_true = (1 - epsilon ** 2) / (1 - epsilon * np.cos(theta))
    error = np.abs((r_true - r) / r_true)
    return error


# Set initial position and velocity of the comet. Rewritten to use predefined values
r0 = 1.0
v0 = 2 * np.pi
epsilon = 0.5
r = np.array([r0, 0.0])
v = np.array([0.0, v0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines

# Set physical parameters
GM = 4 * np.pi ** 2  # Gravitational constant * Mass of sun [AU^3/yr^2]
mass = 1.0  # Mass of comet (reference mass)
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Loop over the desired number of steps using Euler-Cromer and Verlet Methods.
nStep = 1000

# Allows user to pick from three time steps
tau_list = eval(input('Choose timesteps: 1) 0.01; 2) 0.005; 3) 0.001  '))
NumericalMethod = [1, 2]

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)
kinetic = np.empty(nStep)
potential = np.empty(nStep)
error = np.empty(nStep)

# Rewritten to use two methods we are concerned with
for num in NumericalMethod:

    NumericalMethod = num

    for iStep in tqdm(range(nStep)):

        # Record position and energy for plotting
        rplot[iStep] = np.linalg.norm(r)  # Record radial position and angle for polar plot
        thplot[iStep] = np.arctan2(r[1], r[0])
        tplot[iStep] = time
        kinetic[iStep] = 0.5 * mass * np.linalg.norm(v) ** 2  # Record kinetic and potential energy
        potential[iStep] = - GM * mass / np.linalg.norm(r)
        error[iStep] = calc_error(rplot[iStep], epsilon=0.6, theta=thplot[iStep]) # Call error function

        # Check for timestep
        if tau_list == 1:
            tau = 0.01
        if tau_list == 2:
            tau = 0.005
        if tau_list == 3:
            tau = 0.001

        # Calculate new position and velocity using the two methods available
        if NumericalMethod == 1:
            methodname = 'EulerCromer'
            accel = -GM * r / np.linalg.norm(r) ** 3
            v += tau * accel
            r += tau * v  # Euler-Cromer Step
            time += tau
        elif NumericalMethod == 2:
            methodname = 'Verlet'
            accel = -GM * r / np.linalg.norm(r) ** 3
            r_half = r + tau / 2 * v
            accel_half = -GM * r_half / np.linalg.norm(r_half) ** 3
            v += tau * (accel + accel_half) / 2
            r += tau * v
            time += tau

    # Graph the trajectory  and energy of the comet over time.
    totalE = kinetic + potential  # total energy

    fig = plt.figure(figsize=(15.0, 5.25))
    ax = fig.add_subplot(131, polar=True)
    ax.plot(thplot, rplot, '+', )
    ax.set_title('Distance (AU)')
    ax.grid(True)
    fig.tight_layout(pad=5.0)

    ax2 = fig.add_subplot(132)
    ax2.plot(tplot, kinetic, ls='-.', label='Kinetic')
    ax2.plot(tplot, potential, ls='--', label='Potential')
    ax2.plot(tplot, totalE, ls='-', label='Total')
    ax2.set_xlabel('Time (yr)')
    ax2.set_ylabel(r'Energy ($M~AU^3/yr^2$)')
    ax2.set_title('Energy Conservation')
    ax2.legend()

    # Plot out error vs time in log space
    ax3 = fig.add_subplot(133)
    ax3.semilogy(tplot, error, label='fractional error')
    ax3.set_xlabel('Time (yr)')
    ax3.set_ylabel('Relative error')
    ax3.set_title('Absolute Fractional Error')

    filename = 'plot_{}_{}.png'.format(methodname, tau)
    fig.savefig(filename)
    plt.show()

'''
14. Modified Adaptive Routine (short): The adaptive Runge-Kutta routine rka uses
a generic method for estimating the error. Write a modified version of rka that accepts
a user-specified function that computes ∆c, the estimate of the truncation error. For
the comet problem, write a function that evaluates the absolute fractional error in the
total energy for the estimate of ∆c. Test your routines and compare with the original
version of rka considered in Figures 3.9 and 3.10 of Garcia.
'''

# Modified version of rka to use the delta_c as the truncation error
def rka_mod(x, t, tau, delta_c_avg, derivsRK, param):
    """
    Adaptive Runge-Kutta routine.
    :param x: Current value of the dependent variable
    :param t: independent variable (usually time)
    :param tau: step size (usually time step)
    :param delta_c_avg: delta_C estimates for the truncation error
    :param derivsRK: right hand side of the ODE; derivsRK is the name of the function which returns dx/dt
    Calling format derivsRK (x, t, param).
    :param param: estra parameters passed to derivsRK
    :return:
    xSmall: New value of the dependent variable
    t: New value of the independent variable
    tau: Suggested step size for next call to rka
    """

    # Set initial variables
    tSave, xSave = t, x  # Save initial values
    safe1, safe2 = 0.9, 4.0  # Safety factors for bounds on tau (hard-coded)
    eps = 1.0e-15

    # Loop over maximum number of attempts to satisfy error bound
    xTemp = np.empty(len(x))
    xSmall = np.empty(len(x))
    xBig = np.empty(len(x))
    maxTry = 100  # Sets a ceiling on the maximum number of adaptive steps

    for iTry in range(maxTry):

        # Take the two small time steps
        half_tau = 0.5*tau
        xTemp = rk4(xSave, tSave, half_tau, derivsRK, param)
        t = tSave + half_tau
        xSmall = rk4(xTemp, t, half_tau, derivsRK, param)

        # Take the one big time step
        t = tSave + tau
        xBig = rk4(xSave, tSave, tau, derivsRK, param)


        # Estimate new tau value (including safety factors)
        tau_old = tau
        tau = safe1 * tau_old * delta_c_avg**(-0.20)
        tau = max(tau, tau_old/safe2)
        tau = min(tau, safe2*tau_old)

        # If error is acceptable, return computed values
        if delta_c_avg < 1:
            return xSmall, t, tau

    # Issue warning message if the error bound is never satisfied.
    print('Warning! Adaptive Runge-Kutta routine failed.')
    return xSmall, t, tau

def delta_c(energies):
    """
    Returns the difference between the total energy, and it's mean
    :param energies: total energy of the system
    :return: returns the delta_c as described in problem 14
    """
    return abs(energies - np.mean(energies))

# Set initial position and velocity of the comet according to values in textbook (Garcia).
r0 = 1.0
v0 = np.pi / 2
r = np.array([r0, 0.0])
v = np.array([0.0, v0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines

# Set physical parameters
GM = 4 * np.pi ** 2  # Gravitational constant * Mass of sun [AU^3/yr^2]
mass = 1.0  # Mass of comet (reference mass)
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Loop over the desired number of steps using the specified numerical method.
nStep = 40
tau = 0.1
NumericalMethod = 1

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)
kinetic = np.empty(nStep)
potential = np.empty(nStep)

for iStep in tqdm(range(nStep)):

    # Record position and energy for plotting
    rplot[iStep] = np.linalg.norm(r)  # Record radial position and angle for polar plot
    thplot[iStep] = np.arctan2(r[1], r[0])
    tplot[iStep] = time
    kinetic[iStep] = 0.5 * mass * np.linalg.norm(v) ** 2  # Record kinetic and potential energy
    potential[iStep] = - GM * mass / np.linalg.norm(r)

    # Calculate new position and velocity using the desired method
    if NumericalMethod == 1:
        state, time, tau = rka(state, time, tau, adaptErr, gravrk, GM)
        r = state[:2]  # 4th Order Runge-Kutta
        v = state[2:]

# Graph the trajectory and energy of the comet over time.
totalE = kinetic + potential  # total energy
delta_c_values = delta_c(totalE) # Estimation of delta_c values for the energy of the system

fig = plt.figure(figsize=(10.0, 5.25))
ax = fig.add_subplot(121, polar=True)
ax.plot(thplot, rplot, '+', )
ax.set_title('Distance (AU)')
ax.grid(True)
fig.tight_layout(pad=5.0)

ax2 = fig.add_subplot(122)
ax2.plot(tplot, kinetic, ls='-.', label='Kinetic')
ax2.plot(tplot, potential, ls='--', label='Potential')
ax2.plot(tplot, totalE, ls='-', label='Total')
ax2.set_xlabel('Time (yr)')
ax2.set_ylabel(r'Energy ($M~AU^3/yr^2$)')
ax2.legend()

plt.show()

# Find the average of the delta_c
avg_delta_c = sum(delta_c_values)/len(delta_c_values)

# Orbit program now using altered error parameter
r0 = 1.0
v0 = np.pi / 2
r = np.array([r0, 0.0])
v = np.array([0.0, v0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines

# Set physical parameters
GM = 4 * np.pi ** 2  # Gravitational constant * Mass of sun [AU^3/yr^2]
mass = 1.0  # Mass of comet (reference mass)
adaptErr = avg_delta_c  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Loop over the desired number of steps using the specified numerical method.
nStep = 40
tau = 0.1
NumericalMethod = 1

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)
kinetic = np.empty(nStep)
potential = np.empty(nStep)

for iStep in tqdm(range(nStep)):

    # Record position and energy for plotting
    rplot[iStep] = np.linalg.norm(r)  # Record radial position and angle for polar plot
    thplot[iStep] = np.arctan2(r[1], r[0])
    tplot[iStep] = time
    kinetic[iStep] = 0.5 * mass * np.linalg.norm(v) ** 2  # Record kinetic and potential energy
    potential[iStep] = - GM * mass / np.linalg.norm(r)

    # Calculate new position and velocity using the desired method
    if NumericalMethod == 1:
        state, time, tau = rka_mod(state, time, tau, avg_delta_c, gravrk, GM) # Using modded rka with avg delta_c
        r = state[:2]  # 4th Order Runge-Kutta
        v = state[2:]

# Graph the trajectory and energy of the comet over time.
totalE = kinetic + potential  # total energy

fig = plt.figure(figsize=(10.0, 5.25))
ax = fig.add_subplot(121, polar=True)
ax.plot(thplot, rplot, '+', )
ax.set_title('Distance (AU)')
ax.grid(True)
fig.tight_layout(pad=5.0)

ax2 = fig.add_subplot(122)
ax2.plot(tplot, kinetic, ls='-.', label='Kinetic')
ax2.plot(tplot, potential, ls='--', label='Potential')
ax2.plot(tplot, totalE, ls='-', label='Total')
ax2.set_xlabel('Time (yr)')
ax2.set_ylabel(r'Energy ($M~AU^3/yr^2$)')
ax2.legend()

plt.show()
