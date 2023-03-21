import numpy as np
import scipy.linalg as sl
import time
from numba import jit, njit

# Using numba to compile and speed up the code
@njit
def generate_circuit_system(size, high_voltage):
    """
    Takes the circuit system size (number of different voltages)
    and returns both the banded form of the circuit system matrix
    and the w vector on the other side of the equation
    """

    # define the w_vector, on the rhs of the equation
    w_vec = np.array([v_plus, v_plus] + ([0] * (system_size - 2)))
    
    # define each of the bands of the circuit matrix
    # this uses list construction from base python
    # not sure if this is more efficient than using numpy
    banded_top = [0, 0] + ([-1] * (system_size - 2))
    banded_middle_top = [0] + ([-1] * (system_size - 1))
    banded_diagonal = [3] + ([4] * (system_size - 2)) + [3]
    banded_middle_bottom = ([-1] * (system_size - 1)) + [0]
    banded_bottom = ([-1] * (system_size - 2)) + [0, 0]
    
    # put together into numpy array of banded form
    banded_form = np.array([banded_top, banded_middle_top, banded_diagonal, banded_middle_bottom, banded_bottom])

    return(banded_form, w_vec)

# Using numba to compile and speed up the code
@jit(forceobj = True)
def find_voltages(system_size, v_plus):
    """
    Finds the voltages for the given circuit system with given system size
    and high voltage. Also, this function checks to make sure the 
    voltages are below the high_voltage and above 0
    """

    # Find the circuit matrix and solve for the matricies 
    voltage = sl.solve_banded((2, 2), *generate_circuit_system(system_size, v_plus))

    # Check to make sure that none of the voltages
    # exceed 5 volts or are lower than 0 volts
    max_volt = max(voltage)
    min_volt = min(voltage)
    worked = True
    # Check for failure
    if (max_volt >= v_plus) or (min_volt <= 0):
        worked = False
    
    # Return the voltage answer and whether or not it worked
    return(voltage, worked)


if __name__ == "__main__":
    system_size = eval(input("Choose a system size (Number of Different Voltages): "))
    v_plus = 5 #High Voltage
    
    print(f"Voltages = {find_voltages(system_size, v_plus)[0]}")
    # Try this for every circuit size until the computer can't handle it
    # Set to False on default to not destroy computers
    loop_till_failed = True
    # Current maximum reached is about 30,000,000, didn't fail, but
    # I had to use my computer for something else
    while loop_till_failed:
        # Run the find voltage function, see what happens
        voltage, worked = find_voltages(system_size, v_plus)
        # Check if it works
        if worked:
            print(f"{system_size} worked!")
        else:
            print('-'*100)
            print(f"Failed")
        # Try the next system size
        system_size += 1
