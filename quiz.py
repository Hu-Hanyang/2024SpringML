import numpy as np

y = np.array([1, -1, 0, 2]).T
x_t = np.array([[1, 0, 1, 1], [2, 0, 1, 1], [0, 0, 2,1 ], [1, 1 ,1, 1]])
x = x_t.T


w = np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y)
print(w)
print(np.sqrt(np.sum(w**2)))