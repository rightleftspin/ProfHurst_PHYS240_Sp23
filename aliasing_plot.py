# aliasing_plot: generate an example of aliasing for two sine waves of differing frequency.

# Set up configuration options and special features
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import datetime


if __name__ == '__main__':

    N = 10
    freq_1 = 0.2
    freq_2 = 0.8

    phase_1 = np.pi
    phase_2 = 0.0

    # Generate the data for the time series
    dt = 1   # Time increment
    t = np.arange(N)*dt               # t = [0, dt, 2*dt, ... ], note t = j here compared to Garcia
    fk = np.arange(N)/(N*dt)           # f = [0, 1/(N*dt), ... ], k index

    y1 = np.sin(2*np.pi*t*freq_1 + phase_1)   # Sine wave time series data - 1
    y2 = np.sin(2*np.pi*t*freq_2 + phase_2)   # Sine wave time series data - 2

    # Lets use a finely sampled function to compare to the data, for plotting purposes:
    tmod = np.linspace(0,t[-1],1024)

    ymod1 = np.sin(2*np.pi*tmod*freq_1 + phase_1)
    ymod2 = np.sin(2*np.pi*tmod*freq_2 + phase_2)

    fig1, ax1 = plt.subplots()
    ax1.set_title(r'$f_s = {0:.2f}, \varphi_s = {1:.1f}\pi$'.format(freq_1, phase_1/np.pi))
    ax1.plot(t, y1, 'o', color='red', markersize=10)
    ax1.plot(tmod, ymod1, ls='--', color='red')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')

    fig2, ax2 = plt.subplots()
    ax2.set_title(r'$f_s = {0:.2f}, \varphi_s = {1:.1f}\pi$'.format(freq_2, phase_2/np.pi))
    ax2.plot(t, y2, 'd', color='blue', markersize=10)
    ax2.plot(tmod, ymod2, ls='-.', color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')

    fig3, ax3 = plt.subplots()
    ax3.set_title('Plot Overlay')
    ax3.plot(tmod, ymod1, ls='--', color='red')
    ax3.plot(tmod, ymod2, ls='-.', color='blue')
    ax3.plot(t, y1, 'o', color='red', markersize=10, label=r'{0:.2f}, {1:.1f}$\pi$'.format(freq_1, phase_1/np.pi))
    ax3.plot(t, y2, 'd', color='blue', markersize=10, label=r'{0:.2f}, {1:.1f}$\pi$'.format(freq_2, phase_2/np.pi))
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.legend(frameon=True, title=r'$f_s, \varphi_s$', loc='upper left')

    fig4, ax4 = plt.subplots()
    ax4.set_title(r'Some time series data')
    ax4.plot(t, y1, 'o', color='black', markersize=10)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Amplitude')

    save = True
    if save:
        today = str(datetime.date.today())
        fig_directory = os.path.expanduser('~/figs_out/' + today)
        try:
            os.makedirs(fig_directory)
        except FileExistsError:
            pass

        timeindex = time.strftime("%H%M%S")
        fig1.savefig(fig_directory + '/240-slow-freq' + '-' + str(timeindex) + '.pdf', transparent=False)
        fig2.savefig(fig_directory + '/240-faster-freq' + '-' + str(timeindex) + '.pdf', transparent=False)
        fig3.savefig(fig_directory + '/240-aliasing-overlay' + '-' + str(timeindex) + '.pdf', transparent=False)
        fig4.savefig(fig_directory + '/240-time-series-data' + '-' + str(timeindex) + '.pdf', transparent=False)

    plt.show()