import numpy as np

def initialize_plus_plus(X, k):
    """ Инициализируйте центры кластеров """
    np.random.seed(42)
    centroids = [X[np.random.choice(X.shape[0], size=1, replace=False)][0]]

    for i in range(1, k):
        D = np.array([1e9] * X.shape[0])

        for now in centroids:
            dist = np.linalg.norm(X - now, axis=1) ** 2
            D = np.minimum(D, dist)

        next = np.random.choice(X.shape[0], p=(D/np.sum(D)))
        centroids.append(X[next])

    return np.array(centroids)

def k_means_plus_plus(X, k, max_iters=100, tol=1e-4):
    """ Реализуйте алгоритм K-means++ """
    centroids = initialize_plus_plus(X, k)
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