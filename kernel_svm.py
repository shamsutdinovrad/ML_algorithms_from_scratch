import numpy as np

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    kernel = np.exp(-gamma * (np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :]   - 2 * (X @ Y.T)))

    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel

def kernel_svm(X, y, gamma=0.1, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    """ Реализуйте Kernel SVM """
    np.random.seed(42)
    K = rbf_kernel(X, gamma=gamma)

    alpha = np.array([0.01] * X.shape[0])
    b = 0
    C = 1 / (X.shape[0] * lambda_param)

    for iter in range(n_iters):

        dalpha = 2 * alpha.copy()
        db = 0

        for i in range(X.shape[0]):
            if 1 - y[i] * (np.sum(alpha * y * K[:, i]) + b) > 0:
                dalpha -= C * y[i] * y * K[:, i]
                db -= C * y[i]

        alpha -= learning_rate * dalpha
        b -= learning_rate * db

    return alpha

def predict_kernel_svm(X, X_train, y_train, alpha, gamma=0.1):
    K = rbf_kernel(X, X_train, gamma)
    return np.sign(np.dot(K, alpha * y_train))