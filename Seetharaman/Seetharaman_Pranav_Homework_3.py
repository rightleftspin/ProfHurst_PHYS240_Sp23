# pendul- Program to compute the motion of a simple pendulum using the Euler or Verlet method

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from interp import intrpf
from functools import partial
from scipy.special import ellipkinc

# Select the numerical method to use: Euler or Verlet
NumericalMethod = eval(input('Choose a numerical method (1: Euler, 2: Verlet): '))

switch_methods = [0, 1]
switch_method = eval(input('Choose a method for calculating the switch time\n (0: Interpolation, 1: Crude Fashion): '))
switch_range = 4
switch_granularity = 100
if switch_method not in switch_methods:
    raise ValueError('Please choose a valid method for calculating the switch time.')

# Set initial position and velocity of the pendulum
theta0 = eval(input('Enter inital pendulum angle (in degrees): '))
theta = theta0 * np.pi/180  # Convert angle to radians
omega = 0.0  # Initial velocity is set to zero.

# Set the physical constants and other variables
g_over_L = 1.0  # The constant g/L
time = 0.0  # Initial time
irev = 0  # Use this index to count the number of reversals
tau = eval(input('Enter time step in s: '))

# Take one backward Euler step to start Verlet method
accel = -g_over_L * np.sin(theta)  # Gravitational accleration
theta_old = theta - omega*tau + 0.5*accel*tau**2

# Loop over the desired number of steps with the time step given and numerical method chosen
nstep = eval(input('Enter number of time steps to run: '))
t_plot = np.empty(nstep)
th_plot = np.empty(nstep)
period = np.empty(nstep)  # used to record the period estimates
time_old = 0

for istep in range(nstep):
    # Record angle and time for plotting
    t_plot[istep] = time
    th_plot[istep] = theta * 180 / np.pi  # Convert angle to degrees
    time += tau

    # Compute new position and velocity using Euler or Verlet method
    accel = -g_over_L * np.sin(theta)  # Gravitational acceleration
    if NumericalMethod == 1:
        theta_old = theta  # Save previous angle
        theta += tau*omega  # Euler method
        omega += tau*accel
    else:
        theta_new = 2*theta - theta_old + tau**2 * accel
        theta_old = theta  # Verlet method
        theta = theta_new

    if istep >= 1:
        if switch_method == 0:

            time_last_three = [t_plot[istep - 1], t_plot[istep], time]
            theta_last_three = [th_plot[istep - 1] * np.pi / 180, th_plot[istep] * np.pi / 180, theta]
            intrpf_partial = partial(intrpf, x = time_last_three, y = theta_last_three)
            intrpf_vectorized = np.vectorize(intrpf_partial)
            time_interpolated = np.linspace(time_last_three[0], time_last_three[2], num = switch_granularity)
            theta_interpolated = intrpf_vectorized(time_interpolated)

            theta_sign = np.sign(theta_interpolated)
            theta_switch = np.where(theta_sign[:-1] != theta_sign[1:])[0]

            if theta_switch.size != 0:
                switch_time = (time_interpolated[theta_switch[0]] + time_interpolated[theta_switch[0] + 1]) / 2
                # In order to not double count since we are going back a few theta values
                if not ((time_old - (switch_range * tau)) < switch_time < (time_old + (switch_range * tau))):
                    print('Turning point at time t = {0:.2f}'.format(switch_time))
                    if irev == 0:
                        # If this is just the first sign change, just record the time
                        time_old = switch_time
                    else:
                        period[irev-1] = 2*(switch_time - time_old)
                        time_old = switch_time
                    irev += 1  # Increment the number of reversals by one.

        elif switch_method == 1:
            # Test if the pendulum has passed through theta = 0, if yes, use the time to estimate the period
            if theta*theta_old < 0:  # Test position for sign change
                print('Turning point at time t = {0:.2f}'.format(time))
                if irev == 0:
                    # If this is just the first sign change, just record the time
                    time_old = time
                else:
                    period[irev-1] = 2*(time - time_old)
                    time_old = time
                irev += 1  # Increment the number of reversals by one.

# Estimate the period of oscillation, including the error bar
nPeriod = irev-1
if nPeriod > 1:
    AvePeriod = np.mean(period[0:nPeriod])
    ErrorBar = np.std(period[0:nPeriod])/np.sqrt(nPeriod)
    print(f'Small Angle Approx. Period = {2 * np.pi * np.sqrt(1/g_over_L):.2f} s')
    print(f'Theoretical Period = {4 * np.sqrt(1/g_over_L) * ellipkinc(np.pi / 2, np.sin(theta0 * np.pi / 180) / 2):.2f} s')
    print('Average Period = {0:.2f} +/- {1:.3f} s'.format(AvePeriod, ErrorBar))
else:
    print('Average Period could not be determined in the alloted time')

print('Total runtime = {0:.2f} s'.format(nstep*tau))
# Graph the oscillations as theta versus time
fig, ax = plt.subplots()
ax.plot(t_plot, th_plot, '+')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\theta$ (degrees)')

plt.savefig("pendulum.pdf")
