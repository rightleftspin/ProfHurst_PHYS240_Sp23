# ftdemo - Discrete Fourier transform demonstration program

# Set up configuration options and special features
import numpy as np
import scipy.fftpack as sft
import time
import matplotlib.pyplot as plt
import datetime
import os
import warnings
import copy
warnings.filterwarnings('ignore')


# Define functions to take in integer and float inputs
def dinput(input_text) :
    return int(input(input_text))


def finput(input_text) :
    return float(input(input_text))


if __name__ == '__main__':
    #* Initialize the sine wave time series to be transformed
    N = dinput('Enter the total number of data points: ')
    freq = finput('Enter the frequency of the sine wave: ')
    phase = np.pi * finput('Enter phase of the sine wave (in units of pi): ')

    freq_extra = finput('Enter the frequency of the extra sine wave: ')
    phase_extra = np.pi * finput('Enter phase of the extra sine wave (in units of pi): ')

    # Generate the data for the time series
    dt = 1   # Time increment
    t = np.arange(N)*dt               # t = [0, dt, 2*dt, ... ], note t = j here compared to Garcia
    y = np.sin(2*np.pi*t*freq + phase) + np.sin(2*np.pi*t*freq_extra + phase_extra)   # Sine wave time series
    fk = np.arange(N)/(N*dt)           # f = [0, 1/(N*dt), ... ], k index

    theta_j = (2 * np.pi * freq * t) % (2 * np.pi)
    y_saw = theta_j / (2 * np.pi)

    y_square = np.where(theta_j < np.pi, 1, -1 )


    # Lets use a finely sampled function to compare to the data, for plotting purposes:
    tmod = np.linspace(0,t[-1],1024)
    ymod = np.sin(2*np.pi*tmod*freq + phase) + np.sin(2*np.pi*tmod*freq_extra + phase_extra)
    first_freq = "$f_s, \phi_s$ = {0:.4f}, {1:.2f}$\pi$".format(freq, phase/np.pi)
    text_vals = "$f_s, \phi_s$ = {0:.4f}, {1:.2f}$\pi$, \n $f_e, \phi_e$ = {2:.4f}, {3:.2f}$\pi$".format(freq, phase/np.pi, freq_extra, phase_extra/np.pi)

    #* Compute the transform using the desired method: direct summation or fast Fourier transform (FFT) algorithm.
    Y = np.zeros(N,dtype=complex)
    Y_saw = np.fft.fft(y_saw)
    Y_square = np.fft.fft(y_square)
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

    # Inversing the fft to reconstruct the signal
    inverse_fft_y = np.fft.ifft(Y)
    
    # Apply a high pass filter
    freq_cut = sft.fftfreq(y.size, d=dt)
    filtered_fft = Y.copy()
    filtered_fft_real, filtered_fft_im = (np.real(filtered_fft), np.imag(filtered_fft))
    filtered_fft_real[(freq_cut < freq)] = 0
    filtered_fft_im[(freq_cut < freq)] = 0
    filtered_fft = filtered_fft_real + 1j * filtered_fft_im
    inverse_filter = np.fft.ifft(filtered_fft)


    stopTime = time.time()

    print('Elapsed time = ', stopTime - startTime, ' seconds')

    # power spectrum :
    P = np.abs(Y)**2
    P_filtered = np.abs(filtered_fft)**2

    plt.rcParams.update({'font.size': 10})  # set bigger default font size for plots

    fig1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(5.5, 9.5))
    #* Graph the time series and its transform
    # Top subplot: time axis
    ax1[0].plot(t,y,'o', label='Input data')
    ax1[0].plot(t,inverse_fft_y,'o', label='Unfilterd Reconstruction')
    ax1[0].plot(t,inverse_filter,'b', label='Filtered Reconstruction')
    ax1[0].plot(tmod,ymod,'--',c='C2', label='Sin function')
    ax1[0].set_title('Time series: ' + text_vals,fontsize=12)
    ax1[0].set_xlabel('Time')
    ax1[0].set_ylabel('$y(t)$')
    ax1[0].legend(frameon=True)
    ax1[0].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    # Middle subplot: fourier transform
    ax1[1].plot(fk, np.real(Y),'-o', label='Real')
    ax1[1].plot(fk, np.imag(Y),'--o',mfc='None', label='Imaginary')
    ax1[1].plot(fk, np.real(filtered_fft),'-b', label='Real Filtered')
    ax1[1].plot(fk, np.imag(filtered_fft),'--b',mfc='None', label='Imaginary Filtered')
    ax1[1].legend(frameon=False)
    ax1[1].set_title('Fourier transform',fontsize=12)
    ax1[1].set_xlabel('Frequency')
    ax1[1].set_ylabel('$Y(k)$')
    ax1[1].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    # Bottom subplot: Power Spectrum
    ax1[2].semilogy(fk, P,'-o', label="Unfiltered Power Spectrum")
    ax1[2].semilogy(fk, P_filtered,'-b', label="Filtered Power Spectrum")
    ax1[2].plot([freq,freq],[min(P),max(P)],'--',c='k',label='true $f$')
    ax1[2].set_title('Power spectrum (unnormalized)', fontsize=12)
    ax1[2].set_xlabel('Frequency')
    ax1[2].set_ylabel('Power')
    ax1[2].legend(loc='best')
    ax1[2].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    plt.tight_layout()

    save = True # Set this flag to true if you want to save the plots.
    if save:
        today = str(datetime.date.today())
        fig_directory = os.path.expanduser('./figs_out/' + today)
        try:
            os.makedirs(fig_directory)
        except FileExistsError:
            pass

        timeindex = time.strftime("%H%M%S")
        fig1.savefig(fig_directory + '/240-ft-examples' + '-' + str(timeindex) + '.pdf', transparent=False)

    #plt.show()

    fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 9.5))
    #* Graph the time series and its transform
    # Top subplot: time axis
    ax2[0].plot(t,y_saw,'-o', label='Input data')
    ax2[0].set_title('Sawtooth Time series: ' + first_freq,fontsize=12)
    ax2[0].set_xlabel('Time')
    ax2[0].set_ylabel('$y(t)$')
    ax2[0].legend(frameon=True)
    ax2[0].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    # Middle subplot: fourier transform
    ax2[1].plot(fk, np.real(Y_saw),'-o', label='Real')
    ax2[1].plot(fk, np.imag(Y_saw),'--o',mfc='None', label='Imaginary')
    ax2[1].legend(frameon=False)
    ax2[1].set_title('Fourier transform',fontsize=12)
    ax2[1].set_xlabel('Frequency')
    ax2[1].set_ylabel('$Y(k)$')
    ax2[1].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    plt.tight_layout()

    if save:
        today = str(datetime.date.today())
        fig_directory = os.path.expanduser('./figs_out/' + today)
        try:
            os.makedirs(fig_directory)
        except FileExistsError:
            pass

        timeindex = time.strftime("%H%M%S")
        fig2.savefig(fig_directory + '/240-ft-saw' + '-' + str(timeindex) + '.pdf', transparent=False)

    fig3, ax3 = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 9.5))
    #* Graph the time series and its transform
    # Top subplot: time axis
    ax3[0].plot(t,y_square,'-o', label='Input data')
    ax3[0].set_title('Square Time series: ' + first_freq,fontsize=12)
    ax3[0].set_xlabel('Time')
    ax3[0].set_ylabel('$y(t)$')
    ax3[0].legend(frameon=True)
    ax3[0].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    # Middle subplot: fourier transform
    ax3[1].plot(fk, np.real(Y_square),'-o', label='Real')
    ax3[1].plot(fk, np.imag(Y_square),'--o',mfc='None', label='Imaginary')
    ax3[1].legend(frameon=False)
    ax3[1].set_title('Fourier transform',fontsize=12)
    ax3[1].set_xlabel('Frequency')
    ax3[1].set_ylabel('$Y(k)$')
    ax3[1].tick_params('both', length=6, width = 1.2, which='major', direction='in')

    plt.tight_layout()

    save = True # Set this flag to true if you want to save the plots.
    if save:
        today = str(datetime.date.today())
        fig_directory = os.path.expanduser('./figs_out/' + today)
        try:
            os.makedirs(fig_directory)
        except FileExistsError:
            pass

        timeindex = time.strftime("%H%M%S")
        fig3.savefig(fig_directory + '/240-ft-square' + '-' + str(timeindex) + '.pdf', transparent=False)

