import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numdifftools import Derivative
import sys, os
from functools import partial

from nm4p.rk4 import rk4
from nm4p.rka import rka

print("Please note that ffmpeg is a required dependency of the animation, it will error out and not produce animations if ffmpeg is not installed")
# If you want to run this code without animating, set the following flag to false
animate = True

mH = 1.00794 #Da

D1 = 4.7466
D3 = 1.9668
Re = 1.40083
alpha = 1.04435
beta = 1.000122
delta = 28.2
epsilon = -17.5
kappa = 0.6
lam = 0.65

au = 27.211386 #eV
Emin = -4.746599999999999
Rs = 1.700828745264

def potential(r1, r2):
    # This function was nearly impossible to code, so I took this from the 
    # github page https://github.com/vkrajnak/PK64/blob/main/PK64.py
    # Turns out, the authors who wrote the original paper were really inconsistent in
    # their notation so it makes it really difficult to reproduce their
    # results at all.
    """
    Parameters
    ----------
    conf : ndarray, shape(3,)
        conf=[r1,r2,theta].
    Returns
    -------
    pot : float [eV]
        Potential energy for hydrogen exchange (Porter Karplus, JCP 40 (1964)).
        By minimal energy we understand potential(Re,infty,0) in accordance with existing collinear work.
        Davis, JCP 86 (1987) states the energy of saddle point (1.70083,1.70083,0) of 0.396 eV.
    """
    conf = [r1, r2, 0]
    R1 = conf[0]
    R2 = conf[1]
    R3 = np.sqrt( conf[0]**2+conf[1]**2+2*conf[1]*conf[0]*np.cos(conf[2]) )
    R = np.array([R1,R2,R3])
    Rl = np.roll(R,1)
    Rm = np.roll(R,2)

    zeta = 1+kappa*np.exp(-lam*R)
    S = (1+zeta*R+zeta*zeta*R*R/3)*np.exp(-zeta*R)
    oneE = D1*( np.exp(-2*alpha*(R-Re)) - 2*np.exp(-alpha*(R-Re)) )
    threeE = D3*( np.exp(-2*beta*(R-Re)) + 2*np.exp(-beta*(R-Re)) )
    J = 0.5*(oneE-threeE) \
        + S*S*( 0.5*(oneE+threeE) + delta*( (1+1/Rl)*np.exp(-2*Rl)+(1+1/Rm)*np.exp(-2*Rm) ) )
    Qd = 0.5*( oneE + threeE + S*S*(oneE-threeE) )
    Q = np.sum(Qd, axis=0)

    onemS123 = 1-np.prod(S, axis=0)
    S12mS22 = S[0]*S[0]-S[1]*S[1]
    S22mS32 = S[1]*S[1]-S[2]*S[2]
    S12mS32 = S[0]*S[0]-S[2]*S[2]
    J1m2 = J[0]-J[1]
    J2m3 = J[1]-J[2]
    J1m3 = J[0]-J[2]
    J123 = epsilon*np.prod(S, axis=0)
    QmJ123 = Q-J123

    c1 = onemS123**2 - 0.5*( S12mS22**2 + S22mS32**2 + S12mS32**2 )
    c2 = -QmJ123*onemS123 + 0.5*(J1m2*S12mS22 + J2m3*S22mS32 + J1m3*S12mS32)
    c3 = QmJ123*QmJ123 - 0.5*(J1m2**2 + J2m3**2 + J1m3**2)
    pot = (-c2-np.sqrt(c2*c2-c1*c3))/c1
    return pot-Emin

def hhh_system(s, t, param):
    """
    This function takes in the current state of the three hydrogen
    atoms in the Porter-Karplus potential and uses that to
    figure out the updated state for the runge-kutta method
    """
    # Unpack the state
    r, p = s[:2], s[2:]

    # Find velocities
    vel = [((2 * p[0]) - p[1])/mH, ((2 * p[1]) - p[0])/mH]
    # Find the time derivative of the momenta based on the
    # porter-karplus potential energy forumla
    # loosely applying partial derivatives by holding
    # one variable constant
    dudr1 = lambda r1: potential(r1, r[1])
    dudr2 = lambda r2: potential(r[0], r2)
    accel = [-Derivative(dudr1)(r[0]), -Derivative(dudr2)(r[1])]
    
    # return the derivatives of the state vector
    deriv = np.array([vel[0], vel[1], accel[0], accel[1]])
    return deriv

def animate_hsystem(i, r1_vals, r2_vals, h1, h2, h3):
    # Generic animation function for a pendulum, can be
    # used with functools partial to be used in the animation
    # class from matplotlib
    r1, r2 = r1_vals[i], r2_vals[i]
    h1.set_center((0, 0))
    h2.set_center((r1, 0))
    h3.set_center((r2, 0))
    return()

data_dir = "./output/ODE_Project"

# initial values used for the simulation
adaptErr = 1.0E-3  # Error parameter used by adaptive Runge-Kutta
time = 0.0
tau = .0001
nStep = 50

# Initial state variables, starts at the resting radius in the
# hydrogen bond, and the extra hydrogen particle starts at 2 au away
# initial momentum can be added but all these variables need to be 
# finely tuned to keep the system from "exploding"
initial_vel = int(sys.argv[1])
state = np.array([Re, 2, initial_vel, 0])

# Initialize the empty radius arrays and time array
r1plot = np.empty(nStep)
r2plot = np.empty(nStep)
tplot = np.empty(nStep)

# Run the Runga Kutta Simulation
for iStep in tqdm(range(nStep)):
    r1plot[iStep] = state[0]
    r2plot[iStep] = state[1]
    tplot[iStep] = time
    # Update the state, time and time step based on the adaptive runga kutta method
    state, time, tau = rka(state, time, tau, adaptErr, hhh_system, [])

# Creating the output directory
os.makedirs(data_dir, exist_ok=True)
print(f"Output is in {data_dir}")

# Plot the r1 vs r2 plots for the H+H+H system
# this is canonically used to check for reactivity
# in the system
plt.plot(r1plot, r2plot)
plt.suptitle("R1 vs R2 for H+H+H")
plt.title(f"Initial Velocity {initial_vel}au/fs ")
plt.xlabel('R1 Distance (au)')
plt.ylabel('R2 Distance (au)')
plt.savefig(f"{data_dir}/r1vr2_{initial_vel}.pdf")

if animate:
    # The rest of this is animation related code
    # Initialize the figure with specific size and dpi
    # Initialize the general variables for the plot
    plot_size_x, plot_size_y, dpi, frames_per_sec = (4, 4, 200, 20)
    circle_radius = 0.08

    # Initialize the figure with set size, x and y limits
    fig = plt.figure()
    fig.set_size_inches(plot_size_x, plot_size_y, True)
    ax = fig.add_subplot(aspect = 'equal')
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-1, 1)
    
    x1, y1 = 0, 0
    circle1 = ax.add_patch(plt.Circle((x1, y1), circle_radius, fc='r', zorder=3))
    x2, y2 = r1plot[0], 0
    circle2 = ax.add_patch(plt.Circle((x2, y2), circle_radius, fc='r', zorder=3))
    x3, y3 = r2plot[0], 0
    circle3 = ax.add_patch(plt.Circle((x3, y3), circle_radius, fc='b', zorder=3))
    
    # Define the animation function for this specific context
    animate = partial(animate_hsystem, r1_vals = r1plot, r2_vals = r2plot, h1 = circle1, h2 = circle2, h3 = circle3)
    # Declare the interval between frames based on the average spacing between
    # points in the plot lists
    interval = tau * 1000
    # Animate the figure!
    pendulum_animation = animation.FuncAnimation(fig, animate, frames = nStep, interval = interval)
    # Save the animation to a file
    video_writer = animation.FFMpegWriter(fps = frames_per_sec)
    pendulum_animation.save(f"{data_dir}/hhh_{initial_vel}.mp4", writer = video_writer, dpi = dpi)
    plt.close()
