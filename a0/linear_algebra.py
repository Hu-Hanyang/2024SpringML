import numpy as np

A = np.array([[2, 0], [0, 0.5]])
B = np.array([[4, 3], [2, 1]])
C= np.array([[1, -2], [-2, 1]])

# a) Find the inverse of the following matrices
inv_A = np.linalg.inv(A)
print(f"The inverse of the matrix A is {inv_A}.")
inv_B = np.linalg.inv(B)
print(f"The inverse of the matrix B is {inv_B}.")
inv_C = np.linalg.inv(C)
print(f"The inverse of the matrix C is {inv_C}.")

# b) Compute BC and CB
BC = np.matmul(B, C)
print(f"The product of the matrices B and C is {BC}.")
CB = np.matmul(C, B)
print(f"The product of the matrices C and B is {CB}.")

# c) Find the eigenvalues and eigenvectors of the matrix C
eig_C = np.linalg.eig(C)
print(f"The eigenvalues of the matrix C are {eig_C[0]}.")
print(f"The eigenvectors of the matrix C are {eig_C[1]}.")