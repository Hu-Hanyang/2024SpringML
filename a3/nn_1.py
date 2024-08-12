import numpy as np

# Activation function
def ReLU(x):
    return np.maximum(0, x)

def Identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of activate functions
def ReLU_derivative(x):
    return np.where(x <= 0, 0, 1)

def Identity_derivative(x):
    return 1

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Forward pass
def forward_pass(X, W0, W1, W2):
    # Layer 1
    Z1 = np.dot(W0, X)
    h1 = np.vstack((ReLU(Z1),[1])) 

    # Layer 2
    Z2 = np.dot(W1, h1)
    h2 = np.vstack((ReLU(Z2),[1])) 

    # Layer 3
    Z3 = np.dot(W2, h2)
    y_pred = Identity(Z3)
    return Z1, h1, Z2, h2, Z3, y_pred

# Backward pass
def backward_pass(X, Y, W0, W1, W2, learning_rate=0.01):
    Z1, h1, Z2, h2, Z3, y_pred = forward_pass(X, W0, W1, W2)

    # Compute gradients
    dZ3 = y_pred - Y
    dW2 = np.dot(dZ3, h2.T)

    dZ2 = np.dot(W2.T, dZ3) * ReLU_derivative(h2)
    dZ2 = dZ2[:-1, :]
    dW1 = np.dot(dZ2, h1.T)

    dZ1 = np.dot(W1.T, dZ2) * ReLU_derivative(h1)
    dZ1 = dZ1[:-1, :]
    dW0 = np.dot(dZ1, X.T)

    # Update weights
    W2 -= learning_rate * dW2
    W1 -= learning_rate * dW1
    W0 -= learning_rate * dW0

    return W0, W1, W2



if __name__ == "__main__":
    # Initialize random weights
    np.random.seed(0)

    W0 = np.random.randint(-10, 11, size=(3, 3))
    # Divide by 10 to get accuracy of 0.1
    W0 = W0 / 10.0

    W1 = np.random.randint(-10, 11, size=(3, 4))
    # Divide by 10 to get accuracy of 0.1
    W1 = W1 / 10.0


    W2 = np.random.randint(-10, 11, size=(1, 4))
    # Divide by 10 to get accuracy of 0.1  
    W2 = W2 / 10.0

    print("W0: \n", W0)
    print("W1: \n", W1)
    print("W2: \n", W2)


    x = np.array([[1], [1], [1]])

    Z1, h1, Z2, h2, Z3, y_pred = forward_pass(x, W0, W1, W2)
    print("Z1: \n", Z1)
    print("h1: \n", h1)
    print("Z2: \n", Z2)
    print("h2: \n", h2)
    print("Z3: \n", Z3)
    print("y_pred: \n", y_pred)
    


