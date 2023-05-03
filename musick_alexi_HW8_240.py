# Advect - program to solve the advection equation using the various hyperbolic PDE schemes.

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Select the numerical parameters (time step, grid spacing, etc.)
method = eval(input('Choose a numerical method: 1) FTCS; 2) Lax; 3)Lax-Wendroff; 4) Leap-frog: '))
N = eval(input('Enter number of grid points: '))
L = 1.0  # System size
h = L/N  # Grid spacing
c = 1.0  # Wave speed

# Set up plot titles
if method == 1:  ### FTCS Method ###
    plotlabel = 'FTCS'
elif method == 2:  ### Lax Method ###
    plotlabel = 'Lax'
elif method == 3:  ### Lax-Wendroff Method  ###
    plotlabel = 'Lax-Wendroff'
elif method == 4: ### Leap-frog Method ###
    plotlabel = 'Leap-frog'
else:
    raise ValueError('Incorrect index chosen for method. Must choose 1, 2, or 3')

print('Time for wave to move one grid spacing (Courant timestep tc) = {0:.2f}'.format(h/c))

tc = h/c
tau = float(input('Enter time step as a fraction of tc: '))
tau *= tc
coeff = -c*tau/(2.0*h)  # Coefficient used by all schemes
coefflw = 2*coeff**2  # Coefficient used by L-W scheme

print('Wave circles the system in {0:.2f} steps'.format(L/(c*tau)))
nStep = int(input('Enter total number of steps: '))

# Set initial and boundary conditions
sigma = 0.1  # Width of the Gaussian pulses
k_wave = np.pi/sigma  # Wave number of the cosine
x = np.arange(N)*h - L/2  # Coordinates of the grid points

# Set up initial condition to be a Gaussian-cosine pulse
a = np.empty(N)
for i in range(N):
    a[i] = np.cos(k_wave*x[i]) * np.exp(-x[i]**2/(2*sigma**2))

# Use periodic boundary conditions
ip = np.arange(N) + 1
ip[N-1] = 0  # ip  = i+1 with periodic b.c.
im = np.arange(N) - 1
im[0] = N-1  # im = i-1 with periodic b.c.

# Initialize plotting variables
iplot = 1  # Plot counter
nplots = 50  # Desired number of plots
aplot = np.empty((N, nplots))
tplot = np.empty(nplots)
aplot[:, 0] = np.copy(a)  # Record the initial state
tplot[0] = 0  # Record the initial time (t = 0)
plotStep = nStep/nplots + 1

# Loop over the desired number of steps
for iStep in range(nStep):  ## MAIN LOOP ##

    # Compute new values of wave amplitude using FTCS, Lax, or Lax-Wendroff method
    if method == 1:  ### FTCS Method ###
        a[:] = a[:] + coeff*(a[ip] - a[im])
    elif method == 2:  ### Lax Method ###
        a[:] = 0.5*(a[ip] + a[im]) + coeff*(a[ip] - a[im])
    elif method == 3:  ### Lax-Wendroff Method  ###
        a[:] = (a[:] + coeff*(a[ip] - a[im])) + coefflw*(a[ip] + a[im] - 2*a[:])
    elif method == 4: ### Leap-frog Method ###
        if iStep == 0:
            a[:] = 0.5*(a[ip] + a[im]) + coeff*(a[ip] - a[im]) # Lax Method to intialize values
            a_old = np.copy(a) # Store previous values of a
        else: # For subsequent time steps
            a_new = np.zeros(N)
            a_new[:] = a_old[:] + 2.0*coeff*(a[ip] - a[im]) # Approximate the derivative of a
            a_old = np.copy(a)
            a[:] = a_new[:]
    else:
        raise ValueError('Incorrect index chosen for method. Must choose 1, 2, or 3')

    # Periodically record a(t) for plotting
    if (iStep+1) % plotStep < 1:  # Every plot_iter steps record
        aplot[:, iplot] = np.copy(a)
        tplot[iplot] = tau*(iStep+1)
        iplot += 1
        print('{0:g} out of {1:g} steps completed.'.format(iStep, nStep))

# Plot the initial and final states.
fig, ax = plt.subplots()
ax.set_title(r'Advection of wave pulse: ' + plotlabel + ', $\Delta t/t_c = $ {0:.3f}'.format(tau/tc))
ax.plot(x, aplot[:, 0], '-', label='Initial')
ax.plot(x, a, '--', label='Final')
ax.set_xlabel('x')
ax.set_ylabel('Amplitude a(x,t)')
ax.legend()

plt.savefig('plot_1 ' + plotlabel + '_dt{:.3f}.png'.format(tau/tc), dpi=300)

# Plot the total wave amplitude versus position and time
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.set_title(r'Advection of wave pulse: ' + plotlabel + ', $\Delta t/t_c = $ {0:.3f}'.format(tau/tc))
Tp, Xp = np.meshgrid(tplot[0:iplot], x)  # Arrange data into a format suitable for 3D plots.
ax2.plot_surface(Tp, Xp, aplot[:, 0:iplot], rstride=1, cstride=1, cmap=cm.viridis)
ax2.view_init(elev=30., azim=-140.)
ax2.set_ylabel('Position')
ax2.set_xlabel('Time')
ax2.set_zlabel('Amplitude')

plt.savefig('plot_2 ' + plotlabel + '_dt{:.3f}.png'.format(tau/tc), dpi=300)
plt.show()