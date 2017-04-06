import numpy as np

from eclustering.pattern_initialization.anomalous_cluster_p_beta import anomalous_cluster_p_beta
from eclustering.common import minkowski_center
from tests.tools.plot import hold_plot


def imwk_means(data, labels, centroids, weights, p, beta):
    def dist_p_beta(n, k):
        y = data[n]
        c = centroids[k]
        w = weights[k]
        return np.dot((np.abs(y - c) ** p), w ** beta)

    K = len(centroids)
    N = data.shape[0]
    V = data.shape[1]

    yi_to_ck = np.zeros(shape=(N, K))
    change = True

    while change:
        for n in range(N):
            for k in range(K):
                yi_to_ck[n][k] = dist_p_beta(n, k)
        # TODO performance leak
        new_labels = np.apply_along_axis(np.argmin, axis=1, arr=yi_to_ck)
        change = np.array_equal(labels, new_labels)
        labels = new_labels

        # update centroids
        for k in range(K):
            centroids[k] = minkowski_center(data[labels == k], p)

        # update weights
        for k in range(K):
            D = np.sum(np.abs(data[labels == k] - centroids[k]) ** p, axis=0)
            if not D.any():
                weights[k] = np.full(shape=V, fill_value=1 / V)
                continue
            for v in range(V):
                w = 1 / (np.sum((D[v] / D) ** (1 / (p - 1))))
                if np.isnan(w):
                    w = 1
                weights[k][v] = w
    return labels, weights, centroids


np.set_printoptions(precision=2, suppress=True)
if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test6.dat")
    labels, centroids, weights = anomalous_cluster_p_beta(data, p=10, beta=1)
    hold_plot(data, labels)
    labels, centroids, weights = imwk_means(data, labels, centroids, weights, p=3, beta=1)
    print('hold plot')
    hold_plot(data, labels)
