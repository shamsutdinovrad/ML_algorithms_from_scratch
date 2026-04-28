from typing import Callable
import numpy as np

# Имплементация PCA
def get_pca_components(cov_matrix):

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    indexes = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:, indexes]

    return eigenvalues, eigenvectors


# Имплементация Kernel-PCA
def linear_kernel(X, Y=None):
    if Y is None:
        Y = X
    kernel = X @ Y.T
    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel

def poly_kernel(X, Y=None, degree=3, coef=1.):
    if Y is None:
        Y = X
    kernel = (X @ Y.T + coef) ** degree
    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    kernel = np.exp(-gamma * (np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)[np.newaxis, :]   - 2 * (X @ Y.T)))

    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel

def sigmoid_kernel(X, Y=None, gamma=None, coef=1.):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    kernel = np.tanh(gamma * (X @ Y.T) + coef)
    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel

def cosine_kernel(X, Y=None, gamma=None, coef=1.):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    kernel = (X @ Y.T) / (np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis]) @ np.sqrt(np.sum(Y**2, axis=1)[np.newaxis, :]))

    assert kernel.shape == (X.shape[0], X.shape[0])
    return kernel


# Центрирование матрицы Грама
def center_kernel(K):
    n = K.shape[0]

    n1 = np.ones((n, n)) / n

    K_new = K - n1 @ K - K @ n1 + n1 @ K @ n1

    return K_new


# Единый класс KernelPCA
class KernelPCA:
    def __init__(self, n_components: int, kernel: Callable, **kernel_params) -> None:
        self.n_components = n_components
        assert isinstance(kernel, Callable), f"kernel should be Callable, you provide {type(kernel)}"
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.eigenvalues = None
        self.eigenvectors = None

    def fit_transform(self, X):
        K = self.kernel(X, **self.kernel_params)
        K_centered = center_kernel(K)
        self.eigenvalues, self.eigenvectors = get_pca_components(K_centered)
        self.eigenvalues = np.clip(self.eigenvalues, a_min=0, a_max=None)
        # Normalize eigenvectors (each column divided by sqrt(eigenvalue))
        self.eigenvectors = self.eigenvectors / np.sqrt(self.eigenvalues + 1e-10)
        self.eigenvalues = self.eigenvalues[:self.n_components]
        return np.dot(K_centered, self.eigenvectors[:, :self.n_components])