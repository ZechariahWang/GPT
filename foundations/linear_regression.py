import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        predictions = X @ weights
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        total_error = 0

        for i in range(len(model_prediction)):
                prediction = float(model_prediction[i][0])
                actual = float(ground_truth[i][0])
                total_error += (prediction - actual) ** 2

        mse = total_error / len(model_prediction)
        return round(mse, 5)
