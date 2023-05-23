# dftcs - Program to solve the diffusion equation using Forward Time Centered Space (FTCS) scheme

# Set up configuration options and special features
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def temp_analytic(n_max, x, t, L, x_0, kappa):
    # Analytical temperature function using method of images
    # Create range to sum over
    sum_range = np.arange(-n_max, n_max + 1)
    t_mesh, x_mesh, n_mesh = np.meshgrid(t, x, sum_range)
    # shift t to sigma squared
    sigma_squared = 2 * kappa * t_mesh
    # find the coefficient
    coeff = 1 / np.sqrt(2 * np.pi * sigma_squared)
    # find the temperature
    temp_unsummed = ((-1) ** np.abs(n_mesh)) * coeff * np.exp(- (((x_mesh + (n_mesh * L)) - x_0) ** 2) / (2 * sigma_squared))
    # Return array summed along the n axis
    return(np.sum(temp_unsummed, axis=2))

# Initialize parameters (time step, grid spacing, etc.)
tau = eval(input('Enter time step: '))
N = eval(input('Enter the number of grid points: '))
L = 1.0  # The system extends from x = -L/2 to L/2
h = L/(N-1)  # Grid size dx
kappa = 1.0  # Diffusion coefficient
coeff = kappa*tau/h**2
t_natural = h**2/(2*kappa)

print('Natural time scale: {0:.2e}'.format(h**2/(2*kappa)))

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

    # Periodically record temperature for plotting.
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

plt.savefig("mesh_grid.pdf")
plt.close()

# Plot temperature versus x and t as a contour plot
fig2, ax2 = plt.subplots()
levels = np.linspace(0.0, 10.0, num=21)
ct = ax2.contour(tplot, xplot, ttplot, levels)
ax2.clabel(ct, fmt='%1.2f')
ax2.set_xlabel('Time')
ax2.set_ylabel('x')
ax2.set_title('Temperature contour plot')

plt.savefig("numerical_contour.pdf")
plt.close()

# Plot analytic temperature vs x and t as a contour plot
analytic_temp = temp_analytic(10, xplot, tplot, L, 0, kappa)
figA, axA = plt.subplots()
ct = axA.contour(tplot, xplot, analytic_temp, levels)
axA.clabel(ct, fmt='%1.2f')
axA.set_xlabel('Time')
axA.set_ylabel('x')
axA.set_title('Analytic Temperature contour plot')

plt.savefig("analytic_contour.pdf")
plt.close()

# Plot Analaytic temp for garcia figure
xplot_extra = np.linspace(-1.5, 1.5, num = 1000)
analytic_temp = temp_analytic(1, xplot_extra, 0.003, L, 0, kappa)
figB, axB = plt.subplots()
axB.plot(xplot_extra, analytic_temp)
axB.set_xlabel('x/L')
axB.set_ylabel('T(x, t)')
axB.set_title('Analytic Temperature via Method of Images at t = 0.03')
axB.vlines(-0.5, -6, 6, linestyles='dashed')
axB.vlines(0.5, -6, 6, linestyles='dashed')
axB.set_ylim([-6, 6])

plt.savefig("analytic_garcia.pdf")
plt.close()

# Plot absolute difference between analytic and numerical solutions
figC, axC =plt.subplots()
temp_diff = np.abs(temp_analytic(10, xplot, tplot, L, 0, kappa) - ttplot)
axC.set_title(r'Absolute Temperature Difference at $\Delta t = {0:.2e}t_a$'.format(tau/t_natural))
axC.plot(xplot, temp_diff[:, 1], label='{0:.2e}'.format(tplot[1]))
axC.plot(xplot, temp_diff[:, 10], label='{0:.2e}'.format(tplot[10]))
axC.plot(xplot, temp_diff[:, 25], label='{0:.2e}'.format(tplot[25]))
axC.plot(xplot, temp_diff[:, -1], label='{0:.2e}'.format(tplot[-1]))
axC.legend(title=r'$t$')
axC.set_xlabel(r'$x$')
axC.set_ylabel(r'$|T_a(x, t) - T_c(x, t)|$')

plt.savefig("1d_slice_diff.pdf")
plt.close()

# Plot 1D slices of the temperature distribution vs. space at short and long times
fig3, ax3 =plt.subplots()
ax3.set_title(r'Bar temperature profile at $\Delta t = {0:.2e}t_a$'.format(tau/t_natural))
ax3.plot(xplot, ttplot[:, 1], label='{0:.2e}'.format(tplot[1]))
ax3.plot(xplot, ttplot[:, 10], label='{0:.2e}'.format(tplot[10]))
ax3.plot(xplot, ttplot[:, 25], label='{0:.2e}'.format(tplot[25]))
ax3.plot(xplot, ttplot[:, -1], label='{0:.2e}'.format(tplot[-1]))
ax3.legend(title=r'$t$')
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$T(x, t)$')

plt.savefig("1d_slice.pdf")
plt.close()
