import numpy as np

def linear_svm(X, y, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    """ Реализуйте Linear SVM """
    np.random.seed(42)

    w = np.array([0.01] * X.shape[1])
    b = 0
    C = 1 / (X.shape[0] * lambda_param)

    for iter in range(n_iters):
        dw = 2 * w.copy()
        db = 0

        for i in range(X.shape[0]):
            if 1 - y[i] * (np.dot(w, X[i]) + b) > 0:
                dw += -C * y[i] * X[i]
                db += -C * y[i]

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

def predict_svm(X, weights, bias):
    return np.sign(np.dot(X, weights) - bias)
