import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"

        z = np.dot(x, w) + b
        
        #activation
        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            y = np.maximum(0, z)

        return np.round(y, 5)