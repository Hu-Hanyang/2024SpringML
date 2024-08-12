import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(x):
    return x[0]**2 + x[1]**2

# Define the constraint function
def constraint(x):
    return x[0] + x[1] - 1

# Initial guess
x0 = np.array([0.5, 0.5])

# Define the bounds for the variables
bounds = ((None, None), (None, None))

# Define the constraint as a dictionary
constraint_dict = {'type': 'eq', 'fun': constraint}

# Solve the optimization problem
result = minimize(objective, x0, bounds=bounds, constraints=constraint_dict)

# Print the result
print("Optimal solution:", result.x)
print("Optimal value:", result.fun)
