# Original Author: Alexi Musick @ Alexi.musick@sjsu.edu

# Homework 2 Phys 240

# Importing packages
import numpy as np
import math

# Problem 1.

'''
1. Orthogonal Vectors (short): Complete the following tasks related to the orthog
program from the test (you can also find a script orthog.py on GitHub).

(a) Modify the orthog program from the textbook so that it accepts a pair of three-
dimensional vectors and outputs a unit vector that is orthogonal to the input
vectors.
(b) Next, modify the program so that if the second vector is not orthogonal to the
first, the program computes a new vector that is orthogonal to the first vector, has
the same length as the second vector, and is in the same plane as the two input
vectors. This orthogonalization is often used with eigenvectors and is commonly
performed using the Gram–Schmidt procedure.
'''


# Part (A):

# Copying the Orthog program from the textbook (Numerical Methods for Physics (Python) by Garcia (pg.22))

# Orthog - Program to test if a pair of vectors is orthogonal. Assumes vectors are in 3D space.

# Function to output a unit vector orthogonal to the the input vectors
def orthogonal_unit_vector(a, b):
    '''
    The function calculates the unit vector orthogonal to the pair of vectors provided by the user.

    return:
        unit_vector: The orthogonal unit vector.
    '''

    # Calculate the cross product of the two vectors
    cross_product = np.cross(a, b)

    # Normalize the cross product to find the unit vector
    unit_vector = cross_product / np.linalg.norm(cross_product)

    # Return the orthogonal unit vector
    return unit_vector


# Ask user to input array elements for vectors A & B
input_a = input('Enter comma-seperated elements for vector A: ')
a = np.fromstring(input_a, sep=',')
input_b = input('Enter comma-seperated elements for vector B: ')
b = np.fromstring(input_b, sep=',')

# Call function to return orthogonal unit vector of A & B
unit_vector = orthogonal_unit_vector(a, b)
print(f'The orthogonal unit vector is:\n{unit_vector}')


# Part (B):

# Function for the Gram-Schmidt procedure
def gram_schmidt(a, b):
    '''
    The function returns a new B vector using the Gram-Schmidt procedure.

    The new vector is orthogonal to vector A and is the same length as the old vector B.

    return:
        b_new: The new B vector that is orthogonal to A and has the sa length as the old B vector.
    '''

    # Check if B is orthogonal to A
    if np.dot(a, b) == 0:
        print(f'The vectors {a} and {b} are orthogonal.')

    # If the b vector is not orthogonal to A compute a new vector with the Gram-Schmidt procedure.

    # Compute the orthogonal component
    proj_b_a = np.dot(b, a) / np.dot(a, a) * a
    orth_b = b - proj_b_a

    # Normalize the new vector to have the same length as the old vector
    norm_orth_b = np.linalg.norm(orth_b)
    norm_b = np.linalg.norm(b)
    b_new = (norm_b / norm_orth_b) * orth_b

    # Return the new B vector
    return b_new


# Ask user to input array elements for vectors A & B
input_a_gram = input('Enter comma-seperated elements for vector A: ')
a_gram = np.fromstring(input_a_gram, sep=',')
input_b_gram = input('Enter comma-seperated elements for vector B: ')
b_gram = np.fromstring(input_b_gram, sep=',')

# Call function to return orthogonal unit vector of A & B
gram_schmidt_vector = gram_schmidt(a_gram, b_gram)
print(f'The new vector is:\n{gram_schmidt_vector}')

# Problem 5.

'''
The probability of flipping N coins and obtaining m “heads"
is given by the binomial distribution to be:

Pn = N!/m!(N-m)!(1/2)^2

What is more probable: flipping 10 coins and getting no heads or flipping 10,000 coins
and getting exactly 5000 heads? Note: You’ll need to create a function to approximate
the factorial for large values.
'''


def log_factorial(n):
    '''
    Computes the logarithm of the factorial using Stirling's approximation.

    return:
        0: if the number of coins is == 0 return 0.
        (math.log(2 * math.pi * n) / 2) + n * math.log(n) - n: if number of coins is not == 0 then do the Stirling' approx. for n coins.
    '''

    # Compute the logarithm of n! using Stirling's approximation
    if n == 0:
        return 0
    else:
        return (math.log(2 * math.pi * n) / 2) + n * math.log(n) - n


def log_binom(n, m):
    '''
    Computes the logarithm of the binomial coefficient.

    return:
        log_factorial(n) - log_factorial(m) - log_factorial(n - m): the logs of the factorial terms in the binomial.
    '''

    # Compute the logarithm of the binomial coefficient
    return log_factorial(n) - log_factorial(m) - log_factorial(n - m)


# Create a function for the binomial distribution:
def coinflip(n, m):
    '''
    Function calculates the probability of getting the exact number of heads (m) in (n) coin flips.

    Will print out probability of getting exact heads (m) for (n) coin flips.

    Try:
        Use the binomial distribution to calculate the probability.

    If N and/or M is large it will cause an overflow.

    Except:
        Use Stirling's approximation to handle large N and M terms.
    '''

    try:
        pn = (((math.factorial(n)) / ((math.factorial(m)) * (math.factorial(n - m)))) * ((1 / 2) ** n))
        print(
            f'The probability of getting exactly {m} heads out of {n} coin flips is \napproximately {pn * 100:.10f} %')
    except:
        log_pn = log_binom(n, m) + n * math.log(0.5)
        pn = math.exp(log_pn)
        print(f"The probability of getting exactly {m} heads out of {n} coin flips is approximately {pn * 100:.10f}%")


# Ask user for input of (n) coin flips and (m) heads
coins = int(input('Number of coins: '))
heads = int(input('Number of heads: '))

# Call coinflip() to calculate the probability of getting (m) heads in (n) coin flips
coinflip(coins, heads)
