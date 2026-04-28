import numpy as np

def k_means(X, k, max_iters=3, tol=1e-4):
    """ Реализуйте алгоритм K-Means """
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], size=k, replace=False)]
    labels = [0] * X.shape[0]

    for iter in range(max_iters):

        centroids_list = [[] for _ in range(k)]

        for i in range(X.shape[0]):
            dist = np.linalg.norm(X[i] - centroids, axis=1)
            nearest = np.argmin(dist)

            centroids_list[nearest].append(i)
            labels[i] = nearest

        old = centroids.copy()

        for j in range(k):
            centroids[j] = np.mean(X[centroids_list[j]], axis=0)

        if np.linalg.norm(old - centroids) < tol:
            break

    return labels, centroids
