import numpy as np


class RLS:
    def __init__(self):
        """
        RLS initialisation
        """

        self.P = np.eye(2)  # state matrix, 2x2
        self.w = np.zeros(
            (2, 1)
        )  # weights (coefficients of the linear model including constant, 2x1)
        self.error = 0

    def update(self, x: float, y: float):
        """
        RLS state update
        :param x: past/input observation (scalar)
        :param y: current/output observation (scalar)
        """

        X = np.reshape([1, x], (1, 2))  # reshape to a 1x2 matrix
        alpha = float(y - X @ self.w)
        g = (self.P @ X.T) / (1 + X @ self.P @ X.T)
        self.error = np.abs(alpha)
        self.w = self.w + g * alpha
        self.P = self.P - g * X * self.P

    def predict(self, x: float) -> float:
        """
        Predict observation, using RLS model
        :param x: past observation (scalar)
        :return: predicted observation (scalar)
        """

        all_zeros = not np.any(self.w)
        if all_zeros:
            return x

        X = np.reshape([1, x], (1, 2))  # reshape to a 1x2 matrix
        return float(X @ self.w)
