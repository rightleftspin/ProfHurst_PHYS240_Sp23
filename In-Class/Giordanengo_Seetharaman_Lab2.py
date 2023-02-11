#!/usr/bin/env python3
import sys
from math import e, factorial
import numpy as np
import matplotlib.pyplot as plt

# Part 1.1
def convert_input_to_float(user_input):
    # Typecase user input to float
    return(float(user_input))

input_to_float = convert_input_to_float(input("Enter a Float: "))
print(input_to_float)

# Part 1.2
print(f"Given Epsilon: {sys.float_info.epsilon}")

exp_tracker = 0.0
eps_exp_func = lambda eps: (10 ** eps) + 1 == 1

while not eps_exp_func(exp_tracker):
    exp_tracker -= .00001

print(f"Empirically-Determined Epsilon (Method 1): {10 ** exp_tracker}")

eps_val = 2e-14
eps_func = lambda eps: eps + 1 == 1

while not eps_func(eps_val):
    eps_val -= 1e-20

print(f"Empirically-Determined Epsilon (Method 2): {eps_val}")

# Part 1.3 a, b, c
def partial_sum(N, X):
    exp_func = lambda n, x: (x ** n) / factorial(n)
    exp_value = 0
    for n in range(N + 1):
        exp_value += exp_func(n, X)
    return(exp_value)

def error_function_good(N, x_input):
    given_e_value = np.array([e ** x_input] * (N + 1))
    empirical_e_value = []
    for n in range(N + 1):
        if x_input >= 0:
            exp_value = partial_sum(n, x_input)
        else:
            exp_value = 1 / partial_sum(n, -x_input)

        empirical_e_value.append(exp_value)

    error = np.abs(empirical_e_value - given_e_value) / given_e_value
    return(error)

def error_function_bad(N, x_input):
    given_e_value = np.array([e ** x_input] * (N + 1))
    empirical_e_value = []
    for n in range(N + 1):
        exp_value = partial_sum(n, x_input)
        empirical_e_value.append(exp_value)

    error = np.abs(empirical_e_value - given_e_value) / given_e_value
    return(error)

N_input = 60
x_input_list = [2, 10, -2, -10]

# Gives bad answers generally for negative results because of alternating between
# adding and subtracting
for x_input in x_input_list:
    error_vals_good = error_function_good(N_input, x_input)
    plt.plot(np.arange(0, N_input + 1), error_vals_good, label = f"{x_input} Good")


error_vals_bad_2 = error_function_bad(N_input, -2)
plt.plot(np.arange(0, N_input + 1), error_vals_bad_2, label = f"{-2} Bad")

error_vals_bad_10 = error_function_bad(N_input, -10)
plt.plot(np.arange(0, N_input + 1), error_vals_bad_10, label = f"{-10} Bad")

plt.legend()
plt.yscale('log')
#plt.xscale('log')
#plt.ylim([0, 1])
plt.xlabel('N values')
plt.ylabel('Log(Error)')
plt.title('Error vs Partial Sum Index')
plt.savefig('error.pdf')
