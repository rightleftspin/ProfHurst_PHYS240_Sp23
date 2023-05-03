# Short 3D plotting demo.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = np.linspace(0,1,50)
y = x**2
z = np.sqrt(x)

fig1 = plt.figure(figsize=(8,8))
ax1 = plt.axes(projection='3d')
ax1.set_box_aspect((1,1,1))
ax1.plot3D(x,y,z,'-o')
ax1.view_init(30,-70) # tilt, rotate angles
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]')
ax1.set_zlabel('$z$ [m]')
ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.set_zlim(0,1)
ax1.tick_params('both', length=10, width=1.2, which='major',labelsize=13)

plt.show()