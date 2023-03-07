# ftdemo - Discrete Fourier transform demonstration program

# Set up configuration options and special features
import numpy as np
import time
import matplotlib.pyplot as plt


# Define functions to take in integer and float inputs
def dinput(input_text) :
    return int(input(input_text))


def finput(input_text) :
    return float(input(input_text))


#* Initialize the sine wave time series to be transformed
N = dinput('Enter the total number of data points: ')
freq = finput('Enter the frequency of the sine wave: ')
phase = np.pi * finput('Enter phase of the sine wave (in units of pi): ')
dt = 1   # Time increment
t = np.arange(N)*dt               # t = [0, dt, 2*dt, ... ], note t = j here compared to Garcia
y = np.sin(2*np.pi*t*freq + phase)   # Sine wave time series
fk = np.arange(N)/(N*dt)           # f = [0, 1/(N*dt), ... ], k index


# lets use a finely sampled function, for plotting purposes:
tmod = np.linspace(0,t[-1],1024)
ymod = np.sin(2*np.pi*tmod*freq + phase)

text_vals = r'$f_s, \phi_s$ = {0:.2f}, {1:.2f}'.format(freq,phase)

#* Compute the transform using the desired method: direct summation or fast Fourier transform (FFT) algorithm.
Y = np.zeros(N,dtype=complex)
Method = dinput('Compute transform by: 1) Direct summation; 2) FFT ? ')

startTime = time.time()
if Method == 1:             # Direct summation
    twoPiN = -2 * np.pi * 1j / N    # (1j) = sqrt(-1)
    for k in range(N):
        for j in range(N):
            expTerm = np.exp(twoPiN*j*k)
            Y[k] += y[j] * expTerm
else:                        # Fast Fourier transform:
    Y = np.fft.fft(y)               # numpy.fft.fft()

stopTime = time.time()

print('Elapsed time = ', stopTime - startTime, ' seconds')

# power spectrum :
P = np.abs(Y)**2

plt.rcParams.update({'font.size': 20})  # set bigger default font size for plots

fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
#* Graph the time series and its transform
# Left subplot: time axis
ax1[0].plot(t,y,'o')
ax1[0].plot(tmod,ymod,'--',c='C2')
ax1[0].set_title('Original time series: ' + text_vals,fontsize=22)
ax1[0].set_xlabel('Time')
ax1[0].set_ylabel('$y(t)$')
ax1[0].tick_params('both', length=8, width=1.2, which='major') # bigger axis ticks

# Right subplot: fourier transform
ax1[1].plot(fk, np.real(Y),'-o', label='Real')
ax1[1].plot(fk, np.imag(Y),'--o',mfc='None', label='Imaginary')
ax1[1].legend(frameon=False)
ax1[1].set_title('Fourier transform: '+ text_vals,fontsize=22)
ax1[1].set_xlabel('Frequency')
ax1[1].set_ylabel('$Y(k)$')
ax1[1].tick_params('both', length=8, width=1.2, which='major') # bigger axis ticks
plt.tight_layout()
#savefig('ftdemo_fig1.png')

#* Compute and graph the power spectrum of the time series
fig2, ax2 = plt.subplots(figsize=(10,8))
ax2.semilogy(fk, P,'-o')
ax2.plot([freq,freq],[min(P),max(P)],'--',c='k',label='true $f$')
ax2.set_title('Power spectrum (unnormalized): ' + text_vals, fontsize=22)
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Power')
ax2.legend(loc='best')
ax2.tick_params('both', length=8, width=1.2, which='major') # bigger axis ticks
#savefig('ftdemo_fig2.png')

plt.show()
