import sys, os, time, json, lmfit
import numpy as np
import sympy as sy
import scipy.linalg as spl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from functools import reduce

from nm4p.rk4 import rk4
from nm4p.rka import rka

# Grab constants from the input json file, makes this code easier to read
constants = json.load(open(sys.argv[1]))
# The constants needed are listed in the json file, which is reasonably self documenting
# to run this code, use
# python Seetharaman_Pranav_Final_Project.py input.json
# It will simulate a basic galactic interaction. Feel free to change
# the input parameters as needed to find different types of
# galactic interactions. It should be decently user friendly.

# Calculated Constants Based on the constants in the input file
constants["Step Range"] = np.arange(0, constants["Maximum Time"] / constants["Time Step"], dtype = int)
constants["Time Tracker"] = np.zeros_like(constants["Step Range"])
constants["Number Objects"] = constants["Number Galaxies"] + (constants["Number Galaxies"] * sum(constants["Star Rings"]))
constants["DOF"] = constants["Number Objects"] * constants["Dimension"] * 2
constants["Masses"] = np.array([[gal_mass] + ([constants["Star Mass (M_solar)"]] * sum(constants["Star Rings"])) for gal_mass in constants["Galactic Mass (M_solar)"]]).flatten()

def galactic_system(state, t, param):
    """
    Inputs:
    state: The current state of the system, every 2 * Dimension values in the array
    corresponds to the positions and velocities of a specific particle

    t: The current time of the system, not used in this function, time independent.

    param: The other parameters used in this function, for this system, we use mass and the
    graviatational constant

    Outputs:
    deriv: The derivatives of the position and velocity of each galactic center and their tracer stars.
    """
    # Unpack the parameters
    masses, G, Dimension = param["Masses"], param["G"], param["Dimension"]
    # Unpack the state
    state_new = np.reshape(state, (state.size // (2 * Dimension), 2 * Dimension))
    positions = state_new[:, :Dimension]
    # Find the relative radius between every particle in the system
    relative_radius = positions[:, np.newaxis, :] - positions
    # Compute the gravitational force (the cubed is because I don't want to find the unit vectors in the next step)
    mag_rel_rad = np.linalg.norm(relative_radius, axis = 2)
    # Weirdly enough, this prevents infinities from arising later
    mag_rel_rad[mag_rel_rad < 0.1] = np.inf
    # Putting it all together into this line
    grav_accel = (G * masses) / (mag_rel_rad ** 3)
    # multiplying direction and magnitude of gravitational force
    total_grav_accel = (-1 * relative_radius * grav_accel[:, :, None]).sum(axis = 1)
    # Seperate into proper derivative bins
    deriv = np.empty(state.size, dtype = state.dtype)
    # Zipper back and forth between the velocities and accelerations to return the proper derivatives
    spots = np.concatenate([np.arange(n, n + Dimension) for n in range(0, deriv.size, 2 * Dimension)])
    # Zipper every three into the velocities section
    deriv[spots] = state_new[:, Dimension:].flatten()
    # Zipper every other three into the acceleration section
    deriv[spots + Dimension] = total_grav_accel.flatten()

    return(deriv)

def state_updater_full(current_state, current_step, func, constants):
    """
    This is a simple wrapper function because updating the current state using the reduce function
    is a bit annoying. Lambda functions don't allow in place assignment
    Takes the current state, step and other information to return the updated state with
    the runge-kutta fourth order function
    """
    # Update the state using the RK4 method
    dt = constants["Time Step"]
    if constants["Method"] == "rka":
        current_state[current_step + 1, :], constants["Time Tracker"][current_step + 1], constants["Time Step"] = rka(current_state[current_step, :],
                                                                                                  constants["Time Tracker"][current_step],
                                                                                                  dt,
                                                                                                  constants["Error Tolerance"],
                                                                                                  func,
                                                                                                  constants)
    elif constants["Method"] == "rk4":
        current_state[current_step + 1, :] = rk4(current_state[current_step, :], current_step * dt, dt, func, constants)
    else:
        raise ValueError("Please choose between rka or rk4 for the Method option")

    return(current_state)

def create_galaxy(constants, mass, pos_offset, vel_offset, angles):
    """
    Create a galaxy based on a central black hole and tracer stars. It centers the galaxy at the offset
    scales based on galaxy_radius, all fractions of this
    """
    star_rings, galactic_radius, step_range = constants["Star Rings"], constants["Galactic Radius"], constants["Step Range"]
    state_initial = np.array(list(pos_offset) + list(vel_offset))
    # Conversion function from polar to cartesian

    rotation_matrix = lambda theta, phi, al: np.array([[1, 0, 0],
                                                   [0, np.cos(theta), -np.sin(theta)],
                                                   [0, np.sin(theta), np.cos(theta)]]) @ np.array([[np.cos(phi), 0, np.sin(phi)],
                                                    [0, 1, 0],
                                                    [-np.sin(phi), 0, np.cos(phi)]]) @ np.array([[np.cos(al), -np.sin(al), 0],
                                                    [np.sin(al), np.cos(al), 0],
                                                    [0, 0, 1]])

    convert = lambda r, theta: np.dot(rotation_matrix(*angles), np.array([r * np.cos(theta), r * np.sin(theta), 0]))
    convert_vel = lambda r, theta: np.dot(rotation_matrix(*angles), np.array([r * np.sin(theta), -r * np.cos(theta), 0]))

    # For each star ring
    for ind, star_count in enumerate(star_rings):
        # Add stars at a given radius determined by the average galactic radius
        radius = (galactic_radius * (ind + 1)) / len(star_rings)
        # Find velocity for circular orbit at that velocity
        velocity = np.sqrt(constants["G"] * mass / radius)
        # Add the object into the state
        for angle in np.linspace(0, 2 * np.pi, star_count):
            # Use the angle and radius to find the x, y and z components to the position and velocity
            state_initial = np.concatenate((state_initial,
                                            np.concatenate((convert(radius, angle) + pos_offset,
                                                            convert_vel(velocity, angle) + vel_offset))))

    # Define the initial state as an empty 2-Dimensional array
    state = np.empty((step_range.size, int(constants["DOF"] / constants["Number Galaxies"])))
    state[0, :] = state_initial

    return(state)

def general_fit_function(function_string, x_data, y_data, guess_parameters):
    """
    This function takes in a function string to be parsed by Sympy's
    expression parser and tries to fit the data given along the 'x'
    independent variables.
    """
    # Use sympy's parser to find the expression from the input string
    parsed_expression = sy.parsing.sympy_parser.parse_expr(function_string)
    parsed_function = sy.lambdify(list(parsed_expression.free_symbols), parsed_expression)

    # Create a model based on this input string with the independent variable x
    parsed_lm_model = lmfit.Model(parsed_function, independent_vars = ['x'])
    # Run the fit of the model to the data starting with the given parameters
    fit_to_model = parsed_lm_model.fit(data = y_data, x = x_data, **guess_parameters)

    return(fit_to_model, parsed_function)

# Initialize as many galaxies as needed and combine them into a joint state
states = []
for mass, galaxy_offset, vel_offset, angles in zip(constants["Galactic Mass (M_solar)"], constants["Galaxy Offsets"], constants["Galaxy Start Velocity"], constants["Galaxy Angles"]):
    states.append(create_galaxy(constants, mass, galaxy_offset, vel_offset, angles))
state = np.concatenate(states, axis = 1)

# Apply the rk4 function over a range of steps
state_updater = lambda current_state, current_step: state_updater_full(current_state, current_step, galactic_system, constants)
# This is where all the calculation is done.
final_state = reduce(state_updater, constants["Step Range"][:-1], state)

# Creating Theta data from the x and y data, y data goes first for arctan2
theta_data = np.arctan2(final_state[:, 0], final_state[:, 1])
r_data = np.sqrt(final_state[:, 1] ** 2 + final_state[:, 0] ** 2)

# Only fitting for one side of the theta data, since fitting both sides
# causes a lot of problems.
theta_data = theta_data[theta_data < 0]
r_data = r_data[:theta_data.size]

# fit to the "linearized version" where the cosine term is removed
kepler_fit = "c_0 / (1 + (c_1 * x))"
# Running the fit function
fit_model, kepler_function = general_fit_function(kepler_fit,
                                                  np.cos(theta_data),
                                                  r_data,
                                                  {"c_0": 18.5, "c_1": 0.9})
# Uncommenting this will check the model's fitting plot, however
# this is done later on the main plot of the code.
#fit_model.plot()
#plt.title("Fitting Kepler's Laws to Numerical Data")
#plt.xlabel("theta")
#plt.ylabel("r (kpc)")
#plt.savefig(f"{constants['data_dir']}/fitted_data.pdf")
#plt.clf()

# Plotting algorithm to plot all the points over all time
positions = np.arange(0, constants["DOF"], 2 * constants["Dimension"])
fig = plt.figure()
ax = fig.add_subplot()
# Plotting the initial state
ax.plot(final_state[0, positions],
        final_state[0, positions + 1],
        'k.')

# Plotting the final state, usually more useful here
ax.plot(final_state[-100, positions],
        final_state[-100, positions + 1],
        'k.')
# Plotting the full numerical solution to the path of the center of masses
ax.plot(final_state[::, 0], final_state[::, 1], 'b', label = "Numerical Sol.")
ax.plot(final_state[::, (sum(constants["Star Rings"]) + 1) * 6],
        final_state[::, ((sum(constants["Star Rings"]) + 1) * 6) + 1],
        'r',
        label = "Numerical Sol.")

# Plotting the approximate analytical solution, only works for elliptical curves so far.
theta = np.linspace(-np.pi, np.pi)
# base on the curve fit done earlier
l, e = fit_model.values["c_0"], fit_model.values["c_1"]
r = l / (1 + e * np.cos(theta))

ax.plot(r * np.sin(theta), r * np.cos(theta), 'b--', label = "Approx. Analytical Sol.")
ax.plot(r * np.sin(theta), -r * np.cos(theta), 'r--', label = "Approx. Analytical Sol.")
ax.legend()

ax.set_aspect("equal")
ax.set_xlabel("x (Kpc)")
ax.set_ylabel("y (Kpc)")
ax.set_title(f"Galactic Flyby Results, Initial")
ax.set_xlim([-100, 100])
ax.set_ylim([-100, 100])
plt.savefig(f"{constants['data_dir']}/galaxy_motion_plot.pdf")

plt.clf()


print(f"Output is in {constants['data_dir']}")

def animate_system(i, final_state, positions, plot_obj):
    # Generic animation function for a galactic system, can be
    # used with functools partial to be used in the animation
    # class from matplotlib

    #update plot object
    plot_obj.set_xdata(final_state[i, positions])
    plot_obj.set_ydata(final_state[i, positions + 1])

    return()

if constants["animate"]:

    positions = np.arange(0, constants["DOF"], 2 * constants["Dimension"])
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_xlabel("x (Kpc)")
    ax.set_ylabel("y (Kpc)")
    ax.set_title(f"Galactic Flyby Animation")
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    # Plotting the initial state
    plot_obj, = ax.plot(final_state[0, positions],
                       final_state[0, positions + 1],
                       'k.')

    # Plotting the full numerical solution to the path of the center of masses
    ax.plot(final_state[::, 0], final_state[::, 1], 'b', label = "Numerical Sol.")
    ax.plot(final_state[::, (sum(constants["Star Rings"]) + 1) * 6],
            final_state[::, ((sum(constants["Star Rings"]) + 1) * 6) + 1],
            'r',
            label = "Numerical Sol.")

    print("Please note that ffmpeg is a required dependency of the animation, it will error out and not produce animations if ffmpeg is not installed")
    # The rest of this is animation related code
    # Initialize the figure with specific size and dpi
    # Initialize the general variables for the plot
    plot_size_x, plot_size_y, dpi, frames_per_sec, interval = (4, 4, 200, 10, .1)

    # Animate the figure!
    pendulum_animation = animation.FuncAnimation(fig, animate_system, frames = int(constants["Step Range"].size // 10), interval = interval, fargs = (final_state[::10, :], positions, plot_obj))

    # Save the animation to a file
    video_writer = animation.FFMpegWriter(fps = frames_per_sec)
    pendulum_animation.save(f"{constants['data_dir']}/galactic_animation.mp4", writer = video_writer, dpi = dpi)
    plt.close()
