import numpy as np
from collections import deque

def DBSCAN(X, eps=1.5, min_pts=3):
    marks = [-1] * X.shape[0]
    types = ['noise'] * X.shape[0]
    cluster = 0

    for i in range(X.shape[0]):
        if marks[i] != -1:
            continue

        dist = np.linalg.norm(X - X[i], axis=1)
        neighbours = np.where(dist <= eps)[0]

        if len(neighbours) < min_pts:
            continue

        cluster += 1
        marks[i] = cluster
        types[i] = 'core'

        queue = deque(neighbours)

        while len(queue) > 0:
            now = queue.popleft()

            if marks[now] != -1:
                continue

            now_dist = np.linalg.norm(X - X[now], axis=1)
            now_neighbours = np.where(now_dist <= eps)[0]

            if len(now_neighbours) >= min_pts:
                marks[now] = cluster
                types[now] = 'core'
                queue.extend(now_neighbours)
            else:
                marks[now] = cluster
                types[now] = 'border'

    return marks, types