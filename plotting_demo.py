import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # decide on the domain you want to plot over
    x = np.linspace(-6*np.pi, 6*np.pi, 500)

    # generate some data
    y = np.sin(x)
    y2 = np.sin(x-np.pi/2)

    # initialize a figure and axis object with a title
    fig, ax = plt.subplots()
    fig.suptitle('This is a sample figure')
    # plot the data
    ax.plot(x, y, color='k', label=r'$\sin(x)$')
    ax.plot(x, y2, color='b', ls='--', label=r'$\sin(x-\pi/2)$')
    # label you axes!
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # add a legend if you would like
    ax.legend(frameon=True, ncols=2, loc="upper right")

    # display the plot
    plt.show()
