# Original Author: Alexi Musick @ alexi.musick@sjsu.edu
# Homework 6
# Project based on the article Physics of Basketball (Peter J. Brancazio)
# Link: https://aapt-scitation-org.libaccess.sjlibrary.org/doi/pdf/10.1119/1.12511

# Import packages
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
hoop_height = 3.05  # Height of basketball hoop (m)
ball_radius = 0.12  # Radius of basketball (m)

# Target point (center of hoop)
x_target = 5.8  # x-coordinate of target point (m)
y_target = 3.05  # y-coordinate of target point (m)

# Initial conditions
h = 1.76  # Height of average NBA basketball player (m)
v0 = 10.0  # Initial velocity of basketball (m/s)


def f(r):
    '''
    Args:
    r (array): A vector representing the state of the particle.

    Returns:
    numpy.ndarray: A numpy array containing the derivative of the input
    '''
    # r = [x, y, vx, vy]
    return np.array([r[2], r[3], 0, -g])


def rk4_step(r, dt):
    """
    Uses the fourth-order Runge-Kutta method to take a single step in the numerical integration of a system of ordinary differential equations.

    Args:
    r (array-like): A vector representing the current state of the system.
    dt (float): The time step used in the integration.

    Returns:
    numpy.ndarray: A numpy array representing the state of the system after taking a single step in the integration.
    """
    k1 = dt * f(r)
    k2 = dt * f(r + 0.5 * k1)
    k3 = dt * f(r + 0.5 * k2)
    k4 = dt * f(r + k3)
    return r + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Calculate launch angle for target point
launch_angle = math.atan(
    (v0 ** 2 - math.sqrt(v0 ** 4 - g * (g * x_target ** 2 + 2 * (y_target - h) * v0 ** 2))) / (g * x_target))
print(f"Launch angle: {launch_angle * 180 / math.pi:.2f} degrees")

# Plot the trajectory of the basketball
v0x = v0 * math.cos(launch_angle)
v0y = v0 * math.sin(launch_angle)
r = np.array([0.0, h, v0x, v0y])
dt = 0.01
t = 0.0
maxstep = 10000
xvals = []
yvals = []

for i in range(maxstep):
    r = rk4_step(r, dt)
    t += dt
    xvals.append(r[0])
    yvals.append(r[1])

    # Terminate loop when basketball reaches target point
    if r[1] < ball_radius and abs(r[0] - x_target) < ball_radius:
        print(f"Hit the target point at x = {x_target:.2f}, y = {y_target:.2f}")
        break

# Plot the position of the basketball hoop and the ground
fig, ax = plt.subplots()
ground = plt.plot([-10, x_target], [0, 0], 'k--', label='Ground')
hoop_poll = plt.plot([x_target, x_target], [0, 4.07], 'k-')
hoop_basket = plt.plot([x_target - 0.46, x_target], [3.05, 3.05], 'k-')
ax.set_xlim([-1, x_target])
ax.set_ylim([0, hoop_height + 1])
ax.set_aspect('equal', adjustable='box')
plt.legend()

# Plot the trajectory of the basketball
plt.plot(xvals, yvals)
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Basketball trajectory')
plt.savefig('basketball_trajectory.png')
plt.show()

# Define target coordinates
x_target = 5.8
y_target = 3.05

# Try to do some pseudo-monte carlo method to get combinations of velocity and theta
v_min = 8.0  # Minimum initial velocity of basketball (m/s)
v_max = 30.0 # Maximum initial velocity of basketball (m/s)
num_v = 22   # Number of initial velocities to try
theta_min = 10.0   # Minimum launch angle (degrees)
theta_max = 80.0   # Maximum launch angle (degrees)
num_theta = 70     # Number of launch angles to try

# Create arrays of initial velocities and launch angles to try
v_list = np.linspace(v_min, v_max, num_v)
theta_list = np.linspace(theta_min, theta_max, num_theta)

# Initialize array to store results
hits_target = np.zeros((num_v, num_theta), dtype=bool)

# Loop through all combinations of initial velocities and launch angles
for i, v0 in enumerate(v_list):
    for j, theta in enumerate(theta_list):
        # Calculate launch angle in radians
        launch_angle = theta * np.pi / 180.0

        # Calculate initial x and y velocity components
        v0x = v0 * np.cos(launch_angle)
        v0y = v0 * np.sin(launch_angle)

        # Initialize position and time arrays
        r = np.array([0.0, h, v0x, v0y])
        dt = 0.01
        t = 0.0
        maxstep = 10000
        xvals = []
        yvals = []

        # Loop through time steps until basketball hits ground
        for step in range(maxstep):
            r = rk4_step(r, dt)
            t += dt
            xvals.append(r[0])
            yvals.append(r[1])

            # Check if basketball hits target point within the radius of the ball
            if r[1] < ball_radius and abs(r[0] - x_target) < ball_radius and abs(r[1] - y_target) < ball_radius:
                hits_target[i, j] = True
                break

        # Print progress
        print(f"Completed {i*num_theta+j+1}/{num_v*num_theta} combinations")

# Find velocities and angles that hit target point
v_hit, theta_hit = np.where(hits_target)
num_hits = len(v_hit)
print(f"Amount of successful combinations: {num_hits}")

# Plot results
plt.scatter(theta_list[theta_hit], v_list[v_hit])
plt.xlabel('Launch angle (degrees)')
plt.ylabel('Initial velocity (m/s)')
plt.title('Combinations that hit target point')
plt.savefig('Shot_combinations.png')
plt.show()