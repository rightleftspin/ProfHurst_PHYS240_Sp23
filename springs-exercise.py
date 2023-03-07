import numpy as np

k = eval(input('Input spring constants as list k = [k1, k2, k3, k4]: '))
Lvec = eval(input('Input rest lengths as list L = [L1, L2, L3, L4]: '))
Lw = eval(input('Enter the distance between walls: '))

# Build the matrix kk and the vector b
k = np.array(k)
L = np.array(Lvec)

kk = np.array([[-k[0]-k[1], k[1], 0.], [k[1], -k[1]-k[2], k[2]], [0.0, k[2], -k[2]-k[3]]])
b = np.array([-k[0]*L[0] + k[1]*L[1], -k[1]*L[1]+k[2]*L[2], -k[2]*L[2]+k[3]*L[3]-k[3]*Lw])

# Note that this program does not assume the walls are immovable
# i.e. if you try k = [1, 1, 1, 0] L = [2, 2, 1, 1] you will find that x3 resides outside of the wall.

try:
    x = np.linalg.inv(kk)@b
except np.linalg.LinAlgError:
    raise np.linalg.LinAlgError('Singular matrix! The position of the masses is not uniquely defined.')

print('Positions of the masses are:')
print(x)
