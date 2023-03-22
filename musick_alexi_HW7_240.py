# Original author: Alexi Musick @ alexi.musick@sjsu.edu

# Import packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Part I

# 1. The data set in the file curve_data.txt was generated from either a polynomial or a
# sine function y(x) = C + A ∗ sin(x). Can you recover the correct function?
# (a) Use the numpy function "loadtxt" to read in the data file and plot the data
# including error bars.
# (b) Fit a polynomial to the data and calculate the χ2 value. You may use an lmfit
# routine or your own routine. Used the reduced χ2 to decide on the polynomial
# degree. Plot the best-fit model curve on top of the data set, including a legend.
# (c) Now, fit the data with a sine function. You will have to figure out a variable
# transformation in order to use a linear fitting model.
# (d) Compare the reduced χ2 between parts b and c. Which is the preferred model?
# Does that agree with what your eye tells you?

# load data from "curve_data.txt" using np.loadtxt()
data = np.loadtxt('curve_data.txt', skiprows=1)

# extract x, y, and dy from the data
x = data[:, 0]
y = data[:, 1]
dy = data[:, 2]

# Fit a polynomial to the data
degree = 2 # Set the degree of polynomial
coeffs = np.polyfit(x, y, degree)
y_fit = np.polyval(coeffs, x)

# Calculate the chi-square value (X^2)
y_err = dy
chi_sqr =  np.sum(((y - y_fit) / y_err) ** 2) # Using equation (y - y_fit) / y_err
deg = len(x) - degree - 1  # degrees of freedom
reduced_chi_sqr = chi_sqr / deg

# Transform variables to fit a sine function
t = np.pi /2 -x # t ranges from 0 to pi
A = np.column_stack((np.sin(t), np.cos(t))) # Creates matrix and stacks the arrays as columns

# fit a linear combination of cos(x) and sin(x) to the data
coeffs_sine = np.linalg.lstsq(A, y, rcond=None)[0] # Fits a linear combination of the cosine and sine functions to the data
y_fit_sine = np.dot(A, coeffs_sine) # Computes the predicted values of y using best-fit coeffs

# Calculate the chi-square value (X^2)
y_err = dy
chi_sqr_sine =  np.sum(((y - y_fit_sine) / y_err) ** 2) # Using equation (y - y_fit) / y_err
deg = len(x) - degree - 1  # degrees of freedom
reduced_chi_sqr_sine = chi_sqr_sine / deg

# (a) plot the bar graph
plt.bar(x, y, yerr=dy, align='center', alpha=0.5)

# set the x-axis and y-axis labels
plt.xlabel('x')
plt.ylabel('y')

# set the title of the graph
plt.title('Bar graph of x vs y')

# display the plot
plt.savefig('part1a.png')
plt.show()

# (b) plot the data nd the best-fit curve
plt.errorbar(x, y, yerr=dy, fmt='o', label='data')
plt.plot(x, y_fit, label='best-fit')

# set the x-axis and y-axis labels
plt.xlabel('x')
plt.ylabel('y')

# set the title of the graph and display the reduced chi-square value
plt.title(f'Polynomial fit of degree {degree}, reduced χ2 = {reduced_chi_sqr:.2f}')
plt.legend()

# display the plot
plt.savefig('part1b.png')
plt.show()

# (c) plot the data and the best-fit sine curve
plt.errorbar(x, y, yerr=dy, fmt='o', label='data')
plt.plot(x, y_fit_sine, label='best-fit')

# set the x-axis and y-axis labels
plt.xlabel('x')
plt.ylabel('y')

# set the title of the graph and display the reduced chi-square value
plt.title(f'Sine fit, reduced χ2 = {reduced_chi_sqr_sine:.2f}')
plt.legend()

# display the plot
plt.savefig('part1c.png')
plt.show()

# (d) answer:
# The reduced chi-square for part (c) is significantly less than for part (b).
# This suggest that the sine function is a better model for the data over the polynomial.
# Visually we can see this to be true as the sine function closely mathches the plotting of the data points.

# 2. Consider the following stock market data of Dow Jones Averages:
# Day 1 2 3 4 5
# DJA 2470 2510 2410 2350 2240
# (a) Assuming constant error bars, fit this data to polynomials from a straight line to
# a quartic (which will exactly fit the five data points).
# (b) Plot these polynomials from day 1 to 6; the sixth day is October 19, 1987, when
# the market dropped 500 points. Comment on what this exercise tells us about the
# limitations of curve fitting.

# Create arrays
days = np.array([1, 2, 3, 4, 5])
dja = np.array([2470, 2510, 2410, 2350, 2240])

# Create coefficents (quartic)
coeffs1 = np.polyfit(days, dja, 1)
coeffs2 = np.polyfit(days, dja, 2)
coeffs3 = np.polyfit(days, dja, 3)
coeffs4 = np.polyfit(days, dja, 4)

# Evaluate polynomials from day 1 to 6
x = np.linspace(1, 6)
y1 = np.polyval(coeffs1, x)
y2 = np.polyval(coeffs2, x)
y3 = np.polyval(coeffs3, x)
y4 = np.polyval(coeffs4, x)

# Plot polynomials
plt.plot(x, y1, label='Linear')
plt.plot(x, y2, label='Quadratic')
plt.plot(x, y3, label='Cubic')
plt.plot(x, y4, label='Quartic')
plt.plot(days, dja, 'o', label='Data')
plt.axvline(x=6, color='r', linestyle='--', label='Oct 19, 1987')
plt.xlabel('Day')
plt.ylabel('DJA')
plt.legend()

# Show plot
plt.savefig('part2.png')
plt.show()

# I think this exercise shows that while curve fitting is a very powerful tool
# it can lead to issues of overfitting as shown by the cubic and quartic fits.
# This is exasporated even further when small data sets are fitted with polynomals of a high degree.

# Part II

# 3. System of blocks condition number
# Consider a system of coupled masses (such as the example shown below) with N − 1 blocks. The spring
# constants are
# (a) k1 = k2 = k3 = . . . = kN = 1
# (b) k1 = 2k2 = 3k3 = . . . = N kN = 1
# (c) k1 = 4k2 = 9k3 = . . . = N 2kN = 1
# (d) k1 = 2k2 = 4k3 = . . . = 2(N −1)kN = 1
# Compute the condition number of K, and plot it as a function of N . Estimate the value of N for which
# cond(K) exceeds 10^12
# x2
# x1


def create_K(N, k):
    """
    Create a stiiffness matrix for the system of masses with given spring constants.

    Arguments:
    N: number of mmasses in the system.
    k: A array of size 'N' containing the spring constants for each mass.

    Returns:
    K: A N x N matrix representing the stiffness matrix of the system.
    """

    # Intialize the matrix
    K = np.zeros((N, N))
    # Populate the matrix
    for i in range(N):
        for j in range(N):
            if i == j:
                K[i][j] = k[N - 1]
            elif i == j + 1 or i == j - 1:
                K[i][j] = -k[min(i, j)]
    return K  # Return  N x N matrix


def plot_cond(N, k):
    """
    Calculate and plot the condition number of the system of coupled masses with given spring constants.

    Arguments:
    N: number of mmasses in the system.
    k: A array of size 'N' containing the spring constants for each mass.

    Returns:
    None
    """

    # Calculate the condition number for each value of N
    cond_list = np.zeros_like(N, dtype=float)
    for i, n in enumerate(tqdm(N, desc='Calculating condition number')):  # add tqdm progress bar
        K = create_K(n, k)
        cond_list[i] = np.linalg.cond(K)

    # Plotting the condition number as a function of N
    plt.plot(N, cond_list)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('cond(K)')
    plt.title('Condition Number of Coupled Masses System')
    plt.show()

    # Find the value of N for which cond(K) exceeds 10^12
    threshold = 10 ** 12
    for i in range(len(N)):
        if cond_list[i] > threshold:
            print(f"Estimated value of N for the case is {N[i]}")
            break

    plt.savefig(f"plot_{i}.png")


# Define the different sets of spring constants to test
N = np.logspace(1, 3, 1000, dtype=int)
k1 = np.ones(N[-1])
k2 = np.concatenate(([1], np.arange(2, N[-1] + 1)))
k3 = np.concatenate(([1], np.arange(4, N[-1] ** 2 + 1, 4)))
k4 = np.concatenate(([1], np.arange(2, 2 * N[-1], 2)))
ks = [k1, k2, k3, k4]

# Comput the condition number for each set of spring constants
for i, k in enumerate(ks):
    print(f"Calculating condition number for case {chr(ord('a') + i)}...")
    plot_cond(N, k)
