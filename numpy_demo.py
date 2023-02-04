import numpy as np

if __name__ == '__main__':
    # define some scalars
    y = 3
    x = -1

    print(x+y)
    print(x*y)

    # define some vectors
    a = np.array((1.0, 2.0, 3.0))  # row vector
    b = np.array(((4.0), (5.0), (6.)))  # column vector
    bt = np.transpose(b)  # row vector that is the transpoe of b

    # define some matrices
    # A = np.array([[1.0, 0., 1.], [0., 1., -1.], [1., 2., 0]])
    # B = np.array([[0., 1., 1.], [2., 3., -1.], [0., 0., 1]])
    # large zero array
    big_array = np.zeros((4, 256, 64, 1024), dtype=float)
    big_array_slice = big_array[:, :, 0, 1] # take all elements in the first 2 dimensions corresponding to the 0 and 1 element of dimensions 3 and 4

    print(a)
    print(A)

    # loops
    p = np.empty((6,2))
    for i in range(6):
        p[i, 0] = i # assign all elements of the first column to be i
        p[i, 1] = i**2 # assign all elements of the second column to be i^2

    print(p)

    # input and output
    z = input('Enter the value of z as a floating point number: ')
    q = eval(input('enter a numpy function: '))
    print(z)

    # z is a string, need to convert to number to print floating point numbers to specific precision 
    z2 = float(z)
    print('z is equal to {0:g}'.format(z2))
    print('z is equal to {0:.2f}'.format(z2))
    print(q)