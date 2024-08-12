import numpy as np

A = np.array([[1, 0], [2, 1], [0, 1]])
B = np.dot(A, A.T)
print(B)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

# # Print the eigenvalues and eigenvectors
# print("Eigenvalues:")
# print(eigenvalues)
# print("\nEigenvectors:")
# print(eigenvectors)



# Calculate SVD
U, S, VT = np.linalg.svd(B)

# # U, S, VT are the three matrices in the SVD decomposition
# # print("U matrix:")
# # print(U)
# # print("\nSingular values (Sigma):")
# # print(np.diag(S))
# # print("\nV transpose matrix:")
# # print(VT)

# # Normalize each column
# U_normalized = U / np.linalg.norm(U, axis=0)
# V_normalized = VT / np.linalg.norm(VT, axis=0)


# # Print the normalized matrix
# # print("Normalized Matrix U:")
# # print(U_normalized)

# # print("\nSingular values (Sigma):")
# # print(np.diag(S))

# # print("\nNormalized Matrix VT:")
# # print(V_normalized)

# # print(-1/np.sqrt(30))
# # print(1/np.sqrt(6))
# # print(2/np.sqrt(5))

# # rebuild_B = np.dot(U, np.dot(np.diag(S), VT))
# # print(rebuild_B)


U_rebuild = np.array([[-2/np.sqrt(30), 1/np.sqrt(5), 2/np.sqrt(6)],
                      [-5/np.sqrt(30), 0, -1/np.sqrt(6)], 
                      [-1/np.sqrt(30), -2/np.sqrt(5), 1/np.sqrt(6)]])

V_rebuild = U_rebuild.T
print(V_rebuild)
sigma = np.diag(S)
B_rebuild = np.dot(U_rebuild, np.dot(sigma, V_rebuild))
print(B_rebuild)


# U = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [np.sqrt(2)/2, -np.sqrt(2)/2]])
# sigma = np.array([[1, 0], [0, 1/2]])
# V_T = np.array([[np.sqrt(3)/2, 1/2], [-1/2, np.sqrt(3)/2]])

# D = np.dot(U, np.dot(sigma, V_T))
# print(D)