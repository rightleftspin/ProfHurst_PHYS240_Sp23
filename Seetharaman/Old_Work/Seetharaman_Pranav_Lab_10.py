import numpy as np
import matplotlib.pyplot as plt

def mb_dist(velocity, temperature):
    """
        Define the maxwell-boltzman distribution in a way that can be vectorized
    """
    # Set relevant physical constants to 1
    m, k_b = 1, 1
    # Define the coefficient outside the exponential term
    coeff = 4 * (velocity ** 2) * np.pi * ((m / (2 * np.pi * k_b * temperature)) ** (3/2))
    # define the exponent
    exponent = (-m * (velocity ** 2)) / (2 * k_b * temperature) 
    
    # return the probabilities
    return(coeff * np.exp(exponent))


def ar_method(number_points, range_mapped, distribution_func):
    """
        Applying the Acceptance-Rejection method to the maxwell boltzman distribution
    """
    # Defining the range of the distribution function
    a, b = range_mapped
    # Define a velocity range
    velocity_range = np.linspace(a, b)
    # Find the maximum probability of the corresponding distribution for
    # the given velocity range
    mb_dist_calculated = distribution_func(velocity_range)
    # Define both R1 and R2 random ranges
    rand_1, rand_2 = np.random.rand(number_points), np.random.rand(number_points)
    # Fix R1 to the input range
    x_try = a + ((b - a) * rand_1)
    # Find the probability of the each x_try point within the range
    prob_x_try = distribution_func(x_try)
    # Find the trial probabilities based on the max probability
    y_try = rand_2 * (1.05 * mb_dist_calculated.max())

    # Vectorize the acceptance condition into a true-false array
    accep_cond = prob_x_try >= y_try
    # split the x_try and y_try points into accepted and rejected arrays
    split_x = (x_try[accep_cond], x_try[~accep_cond])
    split_y = (y_try[accep_cond], y_try[~accep_cond])

    return(split_x, split_y)



"""
Plotting the maxwell boltzman distribution
"""
# Choose a velocity range and calculate the corresponding distribution
velocity = np.linspace(0, 10)
mb_dist_calculated = mb_dist(velocity, 1)

# Plot the disitrubtion with appropriate labels
plt.figure()
plt.title("Maxwell-Boltzmann Distribution at T = 1")
plt.xlabel("Velocity")
plt.ylabel("Probability")
plt.plot(velocity, mb_dist_calculated)
plt.savefig(f"mbdist.pdf")
plt.close()

"""
Applying the acceptance rejection method to find distribution points
"""
# Find accepted and reject points within the distribution
# Set temperature and velocity range
temperature = 1
num_points = 100
vel_range = (0, 5)
velocity = np.linspace(vel_range[0], vel_range[1])
# Calculate theoretical distribution
mb_dist_calculated = mb_dist(velocity, 1)
# define distribution function at specific definition with lambda
dist_func = lambda velocity: mb_dist(velocity, temperature)
# find accepted and rejected x and y points using the ar method
split_x, split_y = ar_method(num_points, vel_range, dist_func)

# Plotting the accepted, rejected, theoretical and histogram of 
# points within the boltzmann distribution
plt.figure()
plt.title(f"Generated Maxwell_Boltzmann Distribution at T = {temperature}")
plt.xlabel("Velocity")
plt.ylabel("Probability")
plt.plot(split_x[0], split_y[0], 'g.', label = "Accepted")
plt.plot(split_x[1], split_y[1], 'r.', label = "Rejected")
plt.hist(split_x[0], density = True, label = "Normalized Histogram of Accepted Points")
plt.plot(velocity, mb_dist_calculated)
plt.legend()
plt.savefig(f"generated_dist_{num_points}.pdf")
plt.close()

"""
Checking the efficiency fraction
"""
# Calculate the efficiency based on the accepted dividied by the total points
efficiency = len(split_x[0]) / num_points
print(f"Efficiency of the Acceptance-Rejection model is {efficiency:.3f} at {num_points} points")

