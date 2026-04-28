import numpy as np


class BaseLoss:

    def func(self, X, y, w):
        return np.sum((X @ w - y) ** 2) / X.shape[0]

    def grad(self, X, y, w):
        return 2 * X.T @ (X @ w - y) / X.shape[0]


class LinearLoss(BaseLoss):

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        w_edit = w.copy()
        w_edit[0] = 0
        return super().func(X, y, w) + self.l2_coef * np.sum(w_edit ** 2)

    def grad(self, X, y, w):
        w_edit = w.copy()
        w_edit[0] = 0
        return super().grad(X, y, w) + 2 * self.l2_coef * w_edit