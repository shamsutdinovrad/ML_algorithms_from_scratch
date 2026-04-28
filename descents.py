import time
import numpy as np
import matplotlib.pyplot as plt


# Градиентный спуск
class GradientDescent(ManualSolver):
    def __init__(self, lr, name) -> None:
        super().__init__(lr, name, True, False)
    def step(self, w: np.ndarray, k: int, grad_f: np.ndarray, hess_f: None) -> np.ndarray:
        return w - self.lr(k) * grad_f


# Heavy ball
class HeavyBall(ManualSolver):
    def __init__(self, lr, beta, init_w, name) -> None:
        super().__init__(lr, name, True, False)
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.w_prev = init_w

    def step(self, w: np.ndarray, k: int, grad_f: np.ndarray, hess_f: None) -> np.ndarray:
        w_new = w - self.lr(k) * grad_f + self.beta(k) * (w - self.w_prev)
        self.w_prev = w.copy()
        return w_new


# NAG
class NAG(ManualSolver):
    def __init__(self, lr, beta, init_w, name) -> None:
        super().__init__(lr, name, False, False)
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.w_prev = init_w
        self.y_prev = init_w

    def step(self, w: np.ndarray, k: int, grad_f: None, hess_f: None) -> np.ndarray:

        w_new = self.y_prev - self.lr(k) * mush_grad(self.y_prev, train_mush_x, train_mush_y)
        y_new = w_new + self.beta(k) * (w_new - w)

        self.w_prev = w.copy()
        self.y_prev = y_new.copy()

        return w_new


# Newton
class Newton(ManualSolver):
    def __init__(self, lr, name) -> None:
        super().__init__(lr, name, True, True)

    def step(self, w: np.ndarray, k: int, grad_f: np.ndarray, hess_f: np.ndarray) -> np.ndarray:
        return w - self.lr(k) * (np.linalg.inv(hess_f) @ grad_f)