from eclustering.pattern_initialization.anomalous_cluster_p_beta import anomalous_cluster_p_beta
import numpy as np
from scipy.spatial import distance as d


def merge_a_ward(data, labels, a, b, distance):
    labels[labels == a] = b  # merge clusters
    # TODO performance leak
    normalize = np.vectorize(lambda x: np.where(np.unique(labels) == x)[0][0])
    labels = normalize(labels)

    # remove old distance entries
    distance = np.delete(distance, a, 0)
    distance = np.delete(distance, a, 1)

    cb = data[labels == b]
    ct_b = np.mean(cb)
    Nb = len(cb)
    for i in range(len(distance)):
        if i == b:
            distance[i][b] = np.inf
            continue
        ca = data[labels == i]
        ct_a = np.mean(ca)
        Na = len(ca)
        distance[i][b] = ((Na * Nb) / (Na + Nb)) * d.sqeuclidean(ct_a, ct_b)
        distance[b][i] = distance[i][b]
    return distance, labels


def a_ward(data, K_star, labels=None):
    if labels is None:
        labels = np.arange(len(data))
    else:  # normalize labels
        normalize = np.vectorize(lambda x: np.where(np.unique(labels) == x)[0][0])
        labels = normalize(labels)
    K = len(np.unique(labels))

    distance = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(a, K):
            if a != b:
                ca = data[labels == a]
                cb = data[labels == b]
                Na, ct_a = len(ca), np.mean(ca, axis=0)
                Nb, ct_b = len(cb), np.mean(cb, axis=0)
                distance[a][b] = ((Na * Nb) / (Na + Nb)) * d.sqeuclidean(ct_a, ct_b)
                distance[b][a] = distance[a][b]
    while K > K_star:
        print('K_star='+str(K_star))
        print('K='+str(K))

        m = np.argmin(distance)
        min_a = m // len(distance)
        min_b = m % len(distance)
        distance, labels = merge_a_ward(data, labels, max(min_a, min_b), min(min_a, min_b), distance)
        K -= 1
    return labels


if __name__ == "__main__":
    from tests.tools.plot import TestObject
    data = TestObject.load_data("ikmeans_test7.dat")
    K_star = 4
    labels, centroids, weights = a_ward(data, K_star)
