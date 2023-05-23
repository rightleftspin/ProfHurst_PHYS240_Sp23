import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from tqdm import tqdm
import sys, os

from nm4p.rk4 import rk4
from nm4p.rka import rka

def pendul(s, t, gkml_array):
    # Unpack the gravity, spring constant, mass and length array
    g, k, m, l = gkml_array

    # Unpack the state
    r, v = s[:2], s[2:]

    # Find the acceleration vector of the bob based on the acceleration formula
    # derived from the lagrangian
    accel = [((r[0] + l) * (v[1] ** 2)) - ((k * r[0]) / m) + (g * np.cos(r[1])), 
             ((-g * np.sin(r[1])) / (l + r[0])) - ((2 * v[0] * v[1]) / (l + r[0]))]

    # return the derivatives of the state vector
    deriv = np.array([v[0], v[1], accel[0], accel[1]])
    return deriv

def animate_pendulum(i, x_vals, y_vals, rod, bob):
    # Generic animation function for a pendulum, can be
    # used with functools partial to be used in the animation
    # class from matplotlib
    x, y = x_vals[i], y_vals[i]
    rod.set_data([0, x], [0, y])
    bob.set_center((x, y))
    return()

data_dir = "./output/HW_4"

# These constants set the natural scales of measurement for the
# motion of the pendulum
grav = 9.81 # Strength of gravitational constant
mass = .1  # Mass of pendulum bob
length = 1 # Resting length of the pendulum
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0

# Input the initial parameters of the module
theta0 = eval(input("Initial Pendulum Angle (deg): "))
spring = eval(input("Spring Constant (N/m): "))
tau = eval(input("Starting Time Step: "))
nStep = eval(input("Number of Total Steps: "))

# Animation decision
to_animate = eval(input("Would you like to produce an animation (1(yes)/0(no)): "))
if to_animate == 1:
    animation_speed = eval(input("Animation Speed (spacing between frames is dt * animation_speed): "))

# Declare the initial state based on the input angle
state = np.array([0, theta0 * np.pi / 180, 0, 0])

rplot = np.empty(nStep)
thplot = np.empty(nStep)
tplot = np.empty(nStep)
tauplot = np.empty(nStep)

# Run the Runga Kutta Simulation
for iStep in tqdm(range(nStep)):
    rplot[iStep] = state[0]
    thplot[iStep] = state[1]
    tplot[iStep] = time
    tauplot[iStep] = tau
    # Update the state, time and time step based on the adaptive runga kutta method
    state, time, tau = rka(state, time, tau, adaptErr, pendul, [grav, spring, mass, length])

# Update the radial plot length to account for the resting length of the pendulum
# then split it into x and y components
rplot = rplot + length
xplot = rplot * np.sin(thplot)
yplot = -rplot * np.cos(thplot)

# Creating the output directory
os.makedirs(data_dir, exist_ok=True)
print(f"Output is in {data_dir}")

# Plot the x and y motion plot of the pendulum
plt.plot(xplot, yplot)
plt.suptitle("Motion Plot of Springy Pendulum")
plt.title(f"k = {spring} N/m, Theta_0 = {theta0}, Mean dt = {np.mean(tauplot):.2e} sec", fontsize = 8)
plt.xlabel('X Distance (m)')
plt.ylabel('Y Distance (m)')
plt.savefig(f"{data_dir}/motion_plot_{spring}_{theta0}.pdf")

print(f"Average time step for k = {spring} N/m is {np.mean(tauplot):.2e} sec")

# Optional animation of the springy pendulum
if to_animate == 1:
    # Initialize the figure with specific size and dpi
    # Initialize the general variables for the plot
    plot_size_x, plot_size_y, dpi, frames_per_sec = (4, 4, 200, 20)
    circle_radius, line_width = (0.08, 3)
    max_x, max_y = (max(xplot) * 2, -min(yplot) * 1.2)

    # Initialize the figure with set size, x and y limits
    fig = plt.figure()
    fig.set_size_inches(plot_size_x, plot_size_y, True)
    ax = fig.add_subplot(aspect = 'equal')
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, 0)

    # Initialize the rod and bob with the starting values
    x0, y0 = xplot[0], yplot[0]
    line, = ax.plot([0, x0], [0, y0], lw=line_width, c='k')
    circle = ax.add_patch(plt.Circle((x0, y0), circle_radius, fc='r', zorder=3))

    # Define the animation function for this specific context
    animate = partial(animate_pendulum, x_vals = xplot, y_vals = yplot, rod = line, bob = circle)
    # Declare the interval between frames based on the average spacing between
    # points in the plot lists
    interval = np.mean(tauplot) * animation_speed
    # Animate the figure!
    pendulum_animation = animation.FuncAnimation(fig, animate, frames = nStep, interval = interval)
    # Save the animation to a file
    video_writer = animation.FFMpegWriter(fps = frames_per_sec)
    pendulum_animation.save(f"{data_dir}/pendulum_{spring}_{theta0}.mp4", writer = video_writer, dpi = dpi)
    plt.close()
