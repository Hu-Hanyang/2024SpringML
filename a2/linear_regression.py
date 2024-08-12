import numpy as np

def least_squaresL1_EV(x, y, Penalty_factor, learning_rate=0.1, n_iterations=10000):
    n_samples, n_features = x.shape
    # Initialize weights
    w = np.zeros(n_features)

    for _ in range(n_iterations):
        # Compute gradients
        #<<TODO#2>> Add your code here
        y_pred = np.dot(x, w)
        d_weights = np.dot(x.T, (y_pred - y)) + Penalty_factor * np.sign(w)

        # Update weights
        w -= learning_rate * d_weights

    return w

def lasso_regression_predict(X, weights):
    return np.dot(X, weights)

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(0)
    X = np.random.rand(100, 5)
    true_weights = np.array([1, 3.0, 4.1, 4, 6.0])
    noise = np.random.randn(100) * 0.1
    y = np.dot(X, true_weights) + noise

    # Fit the Lasso regression model
    learned_weights = least_squaresL1_EV(X, y, Penalty_factor=0.8, learning_rate=0.01, n_iterations=1000)

    # Predict
    y_pred = lasso_regression_predict(X, learned_weights)

    # Print the learned weights
    print("Learned weights:", learned_weights)
