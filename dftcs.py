# dftcs - Program to solve the diffusion equation using Forward Time Centered Space (FTCS) scheme

# Set up configuration options and special features
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# Initialize parameters (time step, grid spacing, etc.)
tau = eval(input('Enter time step: '))
N = eval(input('Enter the number of grid points: '))
L = 1.0  # The system extends from x = -L/2 to L/2
h = L/(N-1)  # Grid size dx
kappa = 1.0  # Diffusion coefficient
coeff = kappa*tau/h**2

if coeff < 0.5:
    print('Solution is expected to be stable.')
else:
    print('Warning! Solution is expected to be unstable. Consider smaller dt or larger dx.')


# Set initial and boundary conditions.
tt = np.zeros(N)  # Initialize temperature to be zero at all points.
tt[int(N/2)] = 1.0/h  # Set initial condition: delta function of high temperature in the center
# The boundary conditions are tt[0] = tt[N-1] = 0

# Set up loop and plot variables.
xplot = np.arange(N)*h - L/2.0  # Record the x scale for plots
iplot = 0  # Counter used to count plots
nstep = 300  # Maximum number of iterations
nplots = 50  # Number of snapshots (plots) to take
plot_step = nstep/nplots  # Number of time steps between plots

# Loop over the desired number of time steps.
ttplot = np.empty((N, nplots))
tplot = np.empty(nplots)

## MAIN LOOP ##
for istep in range(nstep):
    # Compute new temperature using FTCS scheme. All points in space are updated at once.
    # Note that the endpoints (boundary) is not updated.
    tt[1:N-1] = tt[1:N-1] + coeff*(tt[2:N] + tt[0:N-2] - 2*tt[1:N-1])

    # Periodically recod temperature for plotting.
    if (istep + 1) % plot_step < 1:  # record data for plot every plot_step number of steps. Don't record first step.
        ttplot[:, iplot] = np.copy(tt)  # record a copy of tt(i) for plotting
        tplot[iplot] = (istep+1)*tau  # record time for plots
        iplot += 1

# Plot temperature versus x and t as a wire-mesh plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
Tp, Xp = np.meshgrid(tplot, xplot)
ax.plot_surface(Tp, Xp, ttplot, rstride=2, cstride=2, cmap='YlGn')
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_zlabel(r'T(x,t)')
ax.set_title('Diffusion of a delta spike')

# Plot temperature versus x and t as a contour plot
fig2, ax2 = plt.subplots()
levels = np.linspace(0.0, 10.0, num=21)
ct = ax2.contour(tplot, xplot, ttplot, levels)
ax2.clabel(ct, fmt='%1.2f')
ax2.set_xlabel('Time')
ax2.set_ylabel('x')
ax2.set_title('Temperature contour plot')

plt.show()
