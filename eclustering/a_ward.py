from eclustering.pattern_init import a_pattern_init_p_beta

__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
import itertools
from eclustering.common import minkowski_center, weights_function

np.set_printoptions(precision=3)
p = 0.5
init = False

colors = itertools.cycle(["r", "c", "y", "k"])
markers = itertools.cycle([".", ",", "^", "s"])


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
        m = np.argmin(distance)
        min_a = m // len(distance)
        min_b = m % len(distance)
        distance, labels = merge_a_ward(data, labels, max(min_a, min_b), min(min_a, min_b), distance)
        K -= 1
        # plot(data, labels)
    return labels


def merge_a_ward_p_beta(data, labels, centroids, weights, a, b, distance, p):
    labels[labels == a] = b  # merge clusters
    # TODO performance leak
    normalize = np.vectorize(lambda x: np.where(np.unique(labels) == x)[0][0])
    labels = normalize(labels)

    # remove old entries
    distance = np.delete(distance, a, 0)  # delete old row from distance matrix
    distance = np.delete(distance, a, 1)  # delete old column from distance matrix
    centroids = np.delete(centroids, a, 0)
    weights = np.delete(weights, a, 0)

    cb = data[labels == b]

    # update centroids and weights
    # print(centroids)
    centroids[b] = minkowski_center(cb, p)
    D = np.sum(np.abs(cb - centroids[b]) ** p, axis=0)
    weights[b] = weights_function(D, D=D, p=p)

    ct_b, Nb, wb = centroids[b], len(cb), weights[b]
    for i in range(len(distance)):
        if i == b:
            distance[i][b] = np.inf
            continue
        ca = data[labels == i]
        ct_a, Na, wa = centroids[i], len(ca), weights[i]
        distance[i][b] = ((Na * Nb) / (Na + Nb)) * np.dot(((wa + wb) / 2) ** beta, np.abs(ct_a - ct_b) ** p)
        distance[b][i] = distance[i][b]
    return distance, labels, centroids, weights


def a_ward_p_beta(data, p, beta, K_star, labels, centroids, weights):
    """Assumed that labels starts with 0 and takes ALL values to max value"""
    K = labels.max()
    distance = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(a, K):
            if a != b:
                ca = data[labels == a]
                cb = data[labels == b]
                ct_a, Na, wa = centroids[a], len(ca), weights[a]
                ct_b, Nb, wb = centroids[b], len(cb), weights[b]
                distance[a][b] = ((Na * Nb) / (Na + Nb)) * np.dot(((wa + wb) / 2) ** beta, np.abs(ct_a - ct_b) ** p)
                distance[b][a] = distance[a][b]
    while K > K_star:
        m = np.argmin(distance)
        min_a = m // len(distance)
        min_b = m % len(distance)
        distance, labels, centroids, weights = merge_a_ward_p_beta(data, labels, centroids, weights,
                                                                   max(min_a, min_b), min(min_a, min_b), distance, p)
        K -= 1
    return labels, centroids, weights


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test5.dat")
    from pattern_init import a_pattern_init
    from tests.tools.plot import plot, hold_plot

    # l, c = a_pattern_init(data)
    # l = a_ward(data, 3, l)
    # print(l)
    # hold_plot(data, l)
    p = 3
    beta = 1
    labels, centroids, weights = a_pattern_init_p_beta(data, p, beta)
    labels, centroids, weights = a_ward_p_beta(data, p, beta, 3, labels, centroids, weights)
    hold_plot(data, labels)
