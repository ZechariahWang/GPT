import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        n = X.shape[0] # grabs the number of samples (rows) in your data. X has shape (n_samples, n_features), so .shape is a tuple like (100, 3)
        w = np.zeros(X.shape[1]) # creates the weight vector, initialized to all zeros. X.shape[1] is the number of features (columns), so if you have 3 features, this makes np.array([0.0, 0.0, 0.0]) 
        b = 0.0

        for i in range(epochs):
            y_hat = X @ w + b
            mse = np.mean((y_hat - y)**2)
            dw =  (2/n) * X.T @ (y_hat - y)
            db = (2 / n) * np.sum(y_hat - y)
            w = w - lr * dw
            b = b - lr * db

        return (np.round(w, 5), round(b, 5))
