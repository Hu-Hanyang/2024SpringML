import numpy as np

p = 0.6
H = -p*np.log(p) - (1-p)*np.log(1-p)
print(H)