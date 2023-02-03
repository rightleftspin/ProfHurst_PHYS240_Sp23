#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Sets some constants, grid size and iteration depth
grid_size = 2
granularity = 1000
maximum_iteration_num = 400

# Generate the grid
axis = np.linspace(-grid_size, grid_size, num=granularity, endpoint=True)
x, y = np.meshgrid(axis, axis)
# Create the x and y grid of complex numbers
xy_complex_grid = x + (1j * y)
# Initialize the complex grid for iteration and the iteration number grid
complex_grid = np.zeros_like(xy_complex_grid)
iter_grid_float = np.full_like(xy_complex_grid, maximum_iteration_num, dtype=float)
iter_grid_bool = np.full_like(xy_complex_grid, 0, dtype=int)

current_iteration = 0
# Loop over the specified iteration depth to see if the element is a member
# of the mandelbrot set
while current_iteration <= maximum_iteration_num:
    # Update the complex grid with the mandelbrot equation
    complex_grid = complex_grid ** 2 + xy_complex_grid

    current_iteration += 1

    # lock the iteration count for all
    abs_comp = np.abs(complex_grid)
    for pos, abs_val in np.ndenumerate(abs_comp):
        if abs_val >= 2:
            # checking if |z| is greater than or equal to 2
            # and using this information to update the iteration log
            iter_grid_float[pos[0]][pos[1]] = current_iteration
            iter_grid_bool[pos[0]][pos[1]] = 1

# plotting the multicolor visualization, using imshow because it looks pretty
im = plt.imshow(iter_grid_float, interpolation='bilinear', cmap="jet")
plt.xticks([])
plt.yticks([])
plt.title("Multi-Color Visualization of the Mandelbrot Set")
plt.colorbar(im, label="Number of Iterations")
plt.savefig("mandelbrot_plot_color.png")
plt.close()
# plotting the two color visualization
plt.imshow(iter_grid_bool, interpolation='bilinear', cmap="gray")
plt.xticks([])
plt.yticks([])
plt.title("Two-Color Visualization of the Mandelbrot Set")
plt.savefig("mandelbrot_plot_binary.png")
