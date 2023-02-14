# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:59:46 2023

@author: Matthew
"""

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# we use mpath.Path to determine that the plot we are making is made up of curved segments instead of straight ones
Path = mpath.Path

# mpatches.PathPatch creates a general curve path patch using certain points as well as parameters
# the Path function uses different phrases (Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY) to determine how the graph should be shaped
# moveto moves from one point to the next
# curve3 is a code type built into matplot to create a Bezier curve from the control points
# closepoly creates a line segment from start to finish
fig, ax = plt.subplots()
pp1 = mpatches.PathPatch(
    Path([(0, 0), (2, 1), (-1, 1), (1, 0)],
         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)

plt.show()