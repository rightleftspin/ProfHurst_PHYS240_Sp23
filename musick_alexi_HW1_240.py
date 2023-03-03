# Original Author: Alexi Musick @ Alexi.musick@sjsu.edu

# Homework 1 Phys 240

########################################################################################################################

# Importing packages
import numpy as np
from time import perf_counter

# Problem 2.

'''
2. Investigate the relative efficiency of multiplying two one-dimensional NumPy arrays, a and b. 
You arrays should be large and with non-constant content. Do this in four distinct ways: (a) sum(a*b), (b) np.sum(a*b),
(c) np.dot(a,b), (d) a@b. You are advised to use the default timer() function from the
timeit module to compare the efficiency of different methods. To produce meaningful
results, repeat the calculations several thousands of times (at least).
'''

# Declaring numpy arrays a & b and the respective list alist & blist
alist = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
blist = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
a = np.array(alist)
b = np.array(blist)


# Multiplying one dimensional arrays a & b with four different methods:
def multi_a():
    """
    Method A uses the inbuilt sum method.
    """
    sum(a * b)


def multi_b():
    """
    Method B uses the numpy sum method.
    """
    np.sum(a * b)


def multi_c():
    """
    Method C uses numpy dot product method.
    """
    np.dot(a, b)


def multi_d():
    """
    Method D uses an operator.
    """
    a @ b


# While each method is different they all produce the same result multi_a, multi_b, multi_c, and multi_d = 9650


# Even though each method results in the same answer we want to find which one is more efficient. We do this by using
# the timer() function on each method over several runs (100000) of a while loop.
def time_eff_a():
    """
    The function creates an empty list to be filled by the while loop below.

    The while loop runs over 100000 times to find an accurate efficiency time for the method.
    """

    # Creating an empty list to be filled by the while loop below
    tlist_a = []
    i = 0
    while i < 100000:
        # Time is taken using the time.perf_counter() method
        ta1 = perf_counter()
        # Method is called
        multi_a()
        ta2 = perf_counter()

        # Calculate time from the time.perf_counter() calls
        time = ta2 - ta1
        # Append time to our list
        tlist_a.append(time)

        # Reiterate
        i += 1

    # Average time is calculated over all the runs
    avg_time = sum(tlist_a) / len(tlist_a)

    # Print out average time
    print(avg_time, 'sec')


def time_eff_b():
    """
    The function creates an empty list to be filled by the while loop below.

    The while loop runs over 100000 times to find an accurate efficiency time for the method.
    """

    tlist_b = []
    i = 0
    while i < 100000:
        ta1 = perf_counter()
        multi_b()
        ta2 = perf_counter()

        time = ta2 - ta1

        tlist_b.append(time)

        i += 1

    avg_time = sum(tlist_b) / len(tlist_b)
    print(avg_time, 'sec')


def time_eff_c():
    """
    The function creates an empty list to be filled by the while loop below.

    The while loop runs over 100000 times to find an accurate efficiency time for the method.
    """

    tlist_c = []
    i = 0
    while i < 100000:
        ta1 = perf_counter()
        multi_c()
        ta2 = perf_counter()

        time = ta2 - ta1

        tlist_c.append(time)

        i += 1

    avg_time = sum(tlist_c) / len(tlist_c)
    print(avg_time, 'sec')


def time_eff_d():
    """
    The function creates an empty list to be filled by the while loop below.

    The while loop runs over 100000 times to find an accurate efficiency time for the method.
    """

    tlist_d = []
    i = 0
    while i < 100000:
        ta1 = perf_counter()
        multi_d()
        ta2 = perf_counter()

        time = ta2 - ta1

        tlist_d.append(time)

        i += 1

    avg_time = sum(tlist_d) / len(tlist_d)
    print(avg_time, 'sec')


# We call the functions to find the average times of each method
time_eff_a()
time_eff_b()
time_eff_c()
time_eff_d()

########################################################################################################################

# Problem 9.

'''
The Catalan numbers Cn are a sequence of integers
1,1,2,5,14,42,132... that play an important role in quantum mechanics and the theory
of disordered systems. (They were central to Eugene Wigner’s proof of the so-called
semicircle law.) They are given by:

    C_0 = 1, C_n1 = ((4n+2)/(n+2))Cn

Write a program that prints in increasing order all Catalan numbers less than
or equal to one billion.
'''


# We declare a function that prints out the Catalan numbers in increasing
# order to less than or equal to a billion
def catalan():
    """
    The function writes out the Catalan sequence in increasing order to less

    Than or equal to a billion. This is achieved using a for loop and two IF statements.

    The numbers are printed out in increasing order.
    """

    # Declaring the first term of the Catalan sequence which is C_0 = 1.
    c0 = 1

    # We use a for loop to go through an arbitrarily large range of the sequence
    for n in range(100):

        # For loop begins with an IF statement checking for the start of the sequence
        if n == 0:
            cn = c0
            print('n =', n, int(cn))

        # If the index is not n = 0 then the loop works through the rest of the
        # sequence in increasing order
        cn_1 = ((4 * n + 2) / (n + 2)) * cn
        cn = cn_1

        # IF statement checks if the value of the sequence number is less than or
        # equal to 1 billion. If the number is larger the loop breaks.
        if cn_1 >= 1000000000:
            break

        # Prints out sequence numbers
        print('n =', n + 1, int(cn_1))


# Call Catalan function
catalan()

########################################################################################################################
