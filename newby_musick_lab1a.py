#Importing packages
import numpy as np
import matplotlib.pyplot as plt

#Setting up matrices
A = np.array([[1,2,3],[0,4,5],[0,0,6]])
B = np.array([[1,0,0],[0,2,0],[0,0,3]])
C = np.array([[1,2,3],[4,5,6],[7,8,3]])
D = np.array([[1,1/2,1/3],[1/4,1/5,1/6],[1/7,1/8,1/9]])

#inverse of matrices using numpy linalg.inv()
a_inv = np.linalg.inv(A)
b_inv = np.linalg.inv(B)
c_inv = np.linalg.inv(C)
d_inv = np.linalg.inv(D)

#Printing out matrices, inverse matrices, and the matrix identity for
#Matrix A
print(A)
print(a_inv)
print(np.dot(A,a_inv))
#Printing out matrices, inverse matrices, and the matrix identity for
#Matrix B
print(B)
print(b_inv)
print(np.dot(B,b_inv))
#Printing out matrices, inverse matrices, and the matrix identity for
#Matrix C
print(C)
print(c_inv)
print(np.dot(C,c_inv))
#Printing out matrices, inverse matrices, and the matrix identity for
#Matrix D
print(D)
print(d_inv)
print(np.dot(D,d_inv))

#Generating Eigenvectors and Eigenvalues for matrices
a_eig,a_vector = np.linalg.eig(A)
print(a_eig)
print(a_vector)

vector1 = a_vector[:,0]
vector2 = a_vector[:,1]
vector3 = a_vector[:,2]

eigen1 = a_eig[0]

product1 = np.dot(A,vector1)
product2 = np.dot(eigen1,vector1)
print(product1,product2)

b_eig, b_vector = np.linalg.eig(B)
print(b_eig)
print(b_vector)

vector1 = b_vector[:,0]
vector2 = b_vector[:,1]
vector3 = b_vector[:,2]

eigen1 = b_eig[0]

product1 = np.dot(B,vector1)
product2 = np.dot(eigen1,vector1)
print(product1,product2)

c_eig, c_vector = np.linalg.eig(C)
print(c_eig)
print(c_vector)

vector1 = c_vector[:,0]
vector2 = c_vector[:,1]
vector3 = c_vector[:,2]

eigen1 = c_eig[0]

product1 = np.dot(C,vector1)
product2 = np.dot(eigen1,vector1)
print(product1,product2)

d_eig, d_vector = np.linalg.eig(D)
print(d_eig)
print(d_vector)

vector1 = d_vector[:,0]
vector2 = d_vector[:,1]
vector3 = d_vector[:,2]

eigen1 = d_eig[0]

product1 = np.dot(D,vector1)
product2 = np.dot(eigen1,vector1)
print(product1,product2)

# 2. hydrogen atom

#Declaring Rydberg constant
R = 1.097e-2 #nm^-1

#Wavenumber variable
wavenumber=np.empty(5)

#Lyman series for loop
m=1
for n in range(m+1,m+6,1):
    wavenumber[n-(m+1)]=R*(1/m**2 - 1/((n)**2))
print(wavenumber)

#Balmer series for loop
wavenumber=np.empty(5)
m=2
for n in range(m+1,m+6,1):
    wavenumber[n-(m+1)]=R*(1/m**2 - 1/((n)**2))
print(wavenumber)

#Paschen series for loop
wavenumber=np.empty(5)
m=3
for n in range(m+1,m+6,1):
    wavenumber[n-(m+1)]=R*(1/m**2 - 1/((n)**2))
print(wavenumber)

#3. Plotting practice

#Setting up plot
fig, ax = plt.subplots(figsize=(12,6))
x= np.linspace(0,20,150)

#Defining function
def f(x):
    x = np.exp(-x/4)*np.sin(x)
    return x
ax.plot(x,f(x),color='black')

#Labeling for plot
plt.xlabel('x',size=15)
plt.ylabel('f(x)',size=15)
ax.set_title(r'f(x)=e^(-x/4)sin(x)',size=20)

#Setting limits of plot
plt.xlim([0,20]);plt.ylim([-0.5,1])

#Saving plot as a png file
plt.savefig('sinwave240.png')

#Plt.show() will show plot in terminal
plt.show()