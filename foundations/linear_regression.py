import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        y_pred = np.dot(X, weights)
        # Round to 5 decimal places
        return np.round(y_pred, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        MSE = (np.sum((model_prediction - ground_truth) ** 2)) / len(model_prediction)
        # Round to 5 decimal places
        return np.round(MSE, 5)
        
