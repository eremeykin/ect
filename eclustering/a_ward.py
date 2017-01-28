__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
import itertools

np.set_printoptions(precision=0)
p = 0.5
init = False

colors = itertools.cycle(["r", "c", "y", "k"])
markers = itertools.cycle([".", ",", "^", "s"])


def merge(data, labels, a, b, distance):
    labels[labels == a] = b  # merge clusters
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


def a_ward(data, K, labels=None):
    if labels is None:
        labels = np.arange(len(data))
    else:  # normalize labels
        normalize = np.vectorize(lambda x: np.where(np.unique(labels) == x)[0][0])
        labels = normalize(labels)
    cluster_count = len(np.unique(labels))
    distance = np.full((cluster_count, cluster_count), float("inf"))
    for a in range(cluster_count):
        for b in range(a, cluster_count):
            if a != b:
                ca = data[labels == a]
                cb = data[labels == b]
                Na, ct_a = len(ca), np.mean(ca, axis=0)
                Nb, ct_b = len(cb), np.mean(cb, axis=0)
                distance[a][b] = ((Na * Nb) / (Na + Nb)) * d.sqeuclidean(ct_a, ct_b)
                distance[b][a] = distance[a][b]
    while cluster_count > K:
        m = np.argmin(distance)
        min_a = m // len(distance)
        min_b = m % len(distance)
        distance, labels = merge(data, labels, max(min_a, min_b), min(min_a, min_b), distance)
        cluster_count -= 1
        # plot(data, labels)
    return labels


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test2.dat")
    from pattern_init import a_pattern_init
    from tests.tools.plot import plot, hold_plot
    l, c = a_pattern_init(data)
    l = a_ward(data, 3, l)
    hold_plot(data, l)
