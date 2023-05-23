from math import log, e, pi, sqrt, modf
from gmpy2 import mpz

# The source for many of the formulas used in this code is the wikipedia
# page for double factorials. (https://en.wikipedia.org/wiki/Double_factorial)

def double_factorial_log(n):
    # This function takes in a number and returns the natural log of
    # its double factorial accoring to the product definition of the
    # double factorial. We use gmpy2's mpz object to allow for large
    # number to be used in this function with high precision
    ln_n_double_fac = mpz(0)
    # Check to see if the number is even or odd, to use the appropriate
    # product definition.
    if (n % 2 == 0):
        # Apply the product definition by adding ln(2 * k)
        for k in range(1, ((n // 2) + 1)):
            ln_n_double_fac = ln_n_double_fac + log(2 * k)
    else:
        # Apply the product definition by adding ln(2 * k - 1)
        for k in range(1, (((n + 1) // 2) + 1)):
            ln_n_double_fac = ln_n_double_fac + log(2 * k - 1)

    return(ln_n_double_fac)

def double_factorial_stirlings(n):
    # This function takes in a number and returns the double
    # factorial of it using stirling's approximation.
    # We use the mpz object to account for large numbers.
    n = mpz(n)
    # Check if even or odd
    if (n % 2 == 0):
        # Apply the even version of Stirling's approximation
        n_double_fac = sqrt(pi * n) * ((n / e) ** (n / 2))
    else:
        # Apply the odd version of Stirling's approximation
        n_double_fac = (sqrt(2 * n)) * ((n / e) ** (n / 2))

    return(n_double_fac)

def double_factorial_stirlings_sci(n):
    # This function takes in a nubmer and returns a tuple
    # containing the natural log of the double factorial
    # of the number in scientific notation split into
    # its mantissa and exponent. We use the mpz object
    # for higher precision
    n = mpz(n)
    if (n % 2 == 0):
        # apply a log version of stirling's approximation
        ln_n_double_fac = (log(pi * n) + n * log(n / e)) / (2 * log(10))
    else:
        # apply a log version of stirling's approximation
        ln_n_double_fac = (log(2 * n) + n * log(n / e)) / (2 * log(10))

    # Split the number into it's mantissa and exponent
    n_double_fac_sci = modf(ln_n_double_fac)

    return(n_double_fac_sci)


# Here we print out tests of the various functions for different, large numbers
print(f"Exact: 1000!! = {e ** double_factorial_log(1000):.3E}")
print("-"*100)

print(f"Exact: 2001!! = {e ** double_factorial_log(2001):.3E}")
print("-"*100)

print(f"Exact: 10000!! = {e ** double_factorial_log(10000):.3E}")
print(f"Stirling's Approx: 10000!! = {double_factorial_stirlings(10000):.3E}")
print("-"*100)

print(f"Exact: 314159!! = {e ** double_factorial_log(314159):.3E}")
double_fac_pi = double_factorial_stirlings_sci(314159)
print(f"Stirling's Approx: 314159!! = {10 ** double_fac_pi[0]:.3f}E+{double_fac_pi[1]:3}")
print("-"*100)

double_fac_avag = double_factorial_stirlings_sci(602e21)
print("Could not get more precision than this:")
print(f"Stirling's Approx: {6.02 * (10 ** 23):.3E}!! = {10 ** double_fac_avag[0]:.3f}E+{double_fac_avag[1]:.3E}")
print("-"*100)
