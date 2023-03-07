# ftdemo - Discrete Fourier transform demonstration program

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt


# Define functions for integer and float inputs
def dinput(input_text) :
    return int(input(input_text))


def finput(input_text) :
    return float(input(input_text))


#* Initialize the sine wave time series to be transformed
N = dinput('Enter the number of points: ')
freq = finput('Enter the frequency of the sine wave: ')
phase = np.pi * finput('Enter phase of the sine wave (in units of pi): ')
dt = 1   # Time increment
t = np.arange(N)*dt               # t = [0, dt, 2*dt, ... ], note t = j here
y = np.sin(2*np.pi*t*freq + phase)   # Sine wave time series
fk = np.arange(N)/(N*dt)           # f = [0, 1/(N*dt), ... ], k index