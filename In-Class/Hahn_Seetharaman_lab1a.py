import matplotlib.pyplot as plt
import numpy as np

# Problem (1): Matrix Operations
# Part (a)
# Define our matrices
print("Problem 1:")
A = np.array([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
B = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 3]])
D = np.array([[1, (1/2), (1/3)], [(1/4), (1/5), (1/6)], [(1/7), (1/8), (1/9)]])

matrices = [A, B, C, D]
inv_matrices = []

for matrix in matrices:
    inv_matrices.append(np.linalg.inv(matrix))

multiplied = []
eigen = []
for matrix, inv_matrix in zip(matrices, inv_matrices):
    multiplied.append(np.matmul(matrix, inv_matrix))
    eigen.append(np.linalg.eig(matrix))

# Part (b)
# Printing out matrices with eigenvalues and eigenvectors
for index, matrix in enumerate(matrices):
    print(f"Matrix: {matrix}")
    print(f"Eigenvalues: {eigen[index][0]}")
    print(f"Eigenvectors: {eigen[index][1]}")
    print("-"*50)

# Print out eigenvalue eqn check
# First eigenvector and eigenvalue for matrix B
eigenvec = eigen[1][1][0]
eigenval = eigen[1][0][0]

righthandside = np.matmul(B, eigenvec)
lefthandside = eigenval * eigenvec
print(f"B * x = {righthandside}")
print(f"lambda * x = {lefthandside}")
print("-"*50)


print("Problem 2:")

def Rydberg(n, m):
    R = 1.097e-2
    wavelength = ((n**2) * (m**2)) / ((R)*(n**2 - m**2))
    return(wavelength)

for m in range(1, 4):
    for n in range(m + 1, m + 6):
        print(f"m = {m}, n = {n}: {Rydberg(n, m):.2f} nm")

# Problem 3

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Serif",
    "font.size": 13
})
x = np.linspace(0, 20, 500)
y = np.exp((-x) / 4) * np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y, zorder=5, color="black")
ax.grid(True, zorder=8)

plt.xlim([0, 20])
plt.ylim([-0.4, 1.0])
plt.title("$f(x) = e^{-x/4} sin(x)$")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")

plt.savefig("plot.pdf")
