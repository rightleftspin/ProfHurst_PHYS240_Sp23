# orbit - Program to compute the orbit of a comet.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from nm4p\rk4 import rk4
#from nm4p\rka import rka

def rk4(x, t, tau, derivsRK, param):
    """
    Runge-Kutta integrator (4th order)
    Input arguments
    :param x: current value of dependent variable
    :param t: independent variable (usually time)
    :param tau: step size (usually time step)
    :param derivsRK: right hand side of the ODE; derivsRK is the name of the function which returns dx/dt
    Calling format derivsRK (x, t, param).
    :param param: estra parameters passed to derivsRK
    :return:
    xout: new value of x after a step of size tau
    """

    half_tau = 0.5*tau
    F1 = derivsRK(x, t, param)
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    F2 = derivsRK(xtemp, t_half, param)
    xtemp = x + half_tau*F2
    F3 = derivsRK(xtemp, t_half, param)
    t_full = t + tau
    xtemp = x + tau*F3
    F4 = derivsRK(xtemp, t_full, param)
    xout = x + tau/6.0 * (F1 + F4 + 2.0*(F2+F3))

    return xout

def rka(x, t, tau, err, derivsRK, param):
    """
    Adaptive Runge-Kutta routine.
    :param x: Current value of the dependent variable
    :param t: independent variable (usually time)
    :param tau: step size (usually time step)
    :param err: Desired fractional local truncation error
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

        # Compute the estimated truncation error
        scale = err * 0.5 * (abs(xSmall) + abs(xBig))  # Error times the average of the two quantities
        xDiff = xSmall - xBig
        errorRatio = np.max(np.absolute(xDiff) / (scale + eps))

        # Estimate new tau value (including safety factors)
        tau_old = tau
        tau = safe1 * tau_old * errorRatio**(-0.20)
        tau = max(tau, tau_old/safe2)
        tau = min(tau, safe2*tau_old)

        # If error is acceptable, return computed values
        if errorRatio < 1:
            return xSmall, t, tau

    # Issue warning message if the error bound is never satisfied.
    print('Warning! Adaptive Runge-Kutta routine failed.')
    return xSmall, t, tau



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
    # Setting the alpha constant to be some constant value
    alpha = .005
    # Adjusting the force equation to consider the central force term
    accel = (-GM * r / np.linalg.norm(r)**3) * (1-alpha / np.linalg.norm(r))  # Gravitational acceleration

    # Return derivatives
    deriv = np.array([v[0], v[1], accel[0], accel[1]])

    return deriv


# Set initial position and velocity of the comet.
r0 = 1
v0 = 3
r = np.array([r0, 0.0])
v = np.array([0.0, v0])

state = np.array([r[0], r[1], v[0], v[1]])  # State used by R-K routines

# Set physical parameters
GM = 4 * np.pi**2  # Gravitational constant * Mass of sun [AU^3/yr^2]
mass = 1.0  # Mass of comet (reference mass)
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Loop over the desired number of steps using the specified numerical method.
nStep = 1000
tau = .001
NumericalMethod = 4

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)
kinetic = np.empty(nStep)
potential = np.empty(nStep)
stoplist = []


for iStep in tqdm(range(nStep)):

    # Record position and energy for plotting
    rplot[iStep] = np.linalg.norm(r)  # Record radial position and angle for polar plot
    # print(rplot[iStep])
    thplot[iStep] = np.arctan2(r[1], r[0]) # domain is [-pi, pi]
    tplot[iStep] = time
    kinetic[iStep] = 0.5*mass*np.linalg.norm(v)**2  # Record kinetic and potential energy
    potential[iStep] = - GM*mass/np.linalg.norm(r)

    # Calculate new position and velocity using the desired method
    if NumericalMethod == 1:
        accel = -GM*r/np.linalg.norm(r)**3
        r += tau*v  # Euler Step
        v += tau*accel
        time += tau
    elif NumericalMethod == 2:
        accel = -GM * r / np.linalg.norm(r) ** 3
        v += tau * accel
        r += tau * v  # Euler-Cromer Step
        time += tau
    elif NumericalMethod == 3:
        state = rk4(state, time, tau, gravrk, GM)
        r = state[:2]  # 4th Order Runge-Kutta
        v = state[2:]
        time += tau
    elif NumericalMethod == 4:
        state, time, tau = rka(state, time, tau, adaptErr, gravrk, GM)
        r = state[:2]  # 4th Order Runge-Kutta
        v = state[2:]
    else:
        raise ValueError('Invalid NumericalMethod input. Choose: 1) Euler; 2) Euler-Cromer; 3) Runge-Kutta; 4) Adaptive Runge-Kutta  ')
    # print(np.isclose(thplot[iStep],  3, rtol=1e-2, atol = 1e-2))
    if np.sign(thplot[iStep - 1]) - np.sign(thplot[iStep]) == -2:
        stoplist.append(iStep)

# Graph the trajectory  and energy of the comet over time.
totalE = kinetic + potential  # total energy

fig = plt.figure(figsize=(10.0, 5.25))
ax = fig.add_subplot(121, polar=True)
ax.plot(thplot[:stoplist[0]], rplot[:stoplist[0]], '+',)
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


period = stoplist[0] * tau
halfpind = int(stoplist[0] / 2)
semimajor = (rplot[stoplist[0]] + rplot[halfpind]) / 2

ecc = r0/semimajor - 1
perih = (1 - ecc)*semimajor

## Test program and compare measured eccentricity with predicted value
eTot = np.mean(totalE)
angMom = r0*v0
eccTh = np.sqrt(1 + 2*eTot*angMom**2/(GM**2)*mass**3)

# print(f'Period: {period} AU-yr')
# print(f'Semimajor axis: {semimajor:.3f} AU')
# print(f'Perihelion: {perih:.3f} AU')
# print(f'Eccentricity calculated: {ecc:.3f}')
# print(f'Eccentricity theory: {eccTh:.3f}')

# Setting the alpha constant to be some constant value
alpha = .005
a = np.sqrt(1 + (GM * mass**2 * alpha)/(angMom**2))

print(a)

prec = abs((360 * (1 - a))/a)

print(prec)

