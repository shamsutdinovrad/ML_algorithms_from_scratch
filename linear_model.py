import numpy as np
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w = None
        np.random.seed(random_seed)

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):

        history = {'time': [], 'func': [], 'func_val': []}

        if w_0 is None:
            w_0 = np.array([0.01] * X.shape[1])
        self.w = w_0.copy()

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        for k in range(1, self.max_iter + 1):
            t_0 = time.time()
            w_prev = self.w.copy()

            batch_nums = np.random.choice(X.shape[0], X.shape[0], replace=False)
            for i in range(0, X.shape[0] - self.batch_size + 1, self.batch_size):
                batch_x = X[batch_nums[i:i+self.batch_size]]
                batch_y = y[batch_nums[i:i+self.batch_size]]

                gradient = self.loss_function.grad(batch_x, batch_y, self.w)
                self.w = self.w - gradient * self.step_alpha / (k ** self.step_beta)

            if trace:
                history['time'].append(time.time() - t_0)
                history['func'].append(self.loss_function.func(X, y, self.w))
                if X_val is not None:
                    history['func_val'].append(self.loss_function.func(X_val, y_val, self.w))

            if np.linalg.norm(self.w - w_prev) < self.tolerance:
                break

        if trace:
            return history

    def predict(self, X):
        return X @ self.w

    def get_weights(self):
        return self.w

    def get_objective(self, X, y):
        return self.loss_function.func(X, y, self.w)