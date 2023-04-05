import numpy as np
from nm4p.rk4 import rk4


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