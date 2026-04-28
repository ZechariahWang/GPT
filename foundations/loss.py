import numpy as np
from numpy.typing import NDArray

class Solution:
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        sum_binary=0
        for i in range(len(y_true)):
            p=np.clip(y_pred[i], epsilon, 1-epsilon)
            sum_binary += (y_true[i]*np.log(p)) + (1-y_true[i])*np.log(1-p)
        sum_binary = -sum_binary / len(y_true)

        return np.round(np.array(sum_binary, dtype=np.float64), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        sum_binary=0
        for i in range(len(y_true)):
            for c in range(len(y_true[i])):
                p=np.clip(y_pred[i][c], epsilon, 1-epsilon)
                sum_binary += (y_true[i][c]*np.log(p))
        sum_binary = -sum_binary / len(y_true)

        return np.round(np.array(sum_binary, dtype=np.float64), 4)