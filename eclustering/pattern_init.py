__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
from tests.tools.plot import plot, hold_plot
from eclustering.minkowski import minkowski_center
import collections

import matplotlib

matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt


def a_pattern_init(data):
    original_data = data
    data = np.copy(data)
    indices = np.arange(len(data))
    labels = np.zeros(len(data), dtype=int)
    centroids = []
    origin = np.mean(data, 0)

    def dist_to(y):
        return lambda x: d.sqeuclidean(x, y)

    x_origin = np.apply_along_axis(dist_to(origin), 1, data)
    cluster_label = 1
    while len(data) > 1:
        ct_i = np.argmax(x_origin)
        ct = data[ct_i]
        ct_old = None
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[ct_i] = True
        while not np.array_equal(ct_old, ct):
            ct_old = np.copy(ct)
            x_ct = np.apply_along_axis(dist_to(ct), 1, data)
            anomaly = x_ct < x_origin
            ct = np.mean(data[anomaly], 0)
        normalcy = ~anomaly
        centroids.append(ct)
        data = data[normalcy]
        x_origin = x_origin[normalcy]
        indices = indices[normalcy]
        labels[indices] = cluster_label
        cluster_label += 1
        # plot(original_data, labels)
    centroids.append(data[0])
    return labels, np.array(centroids)


def a_pattern_init_p_beta(data, p, beta):
    def dist_to_p_beta(c, p, beta, w):
        return lambda y: np.dot((np.abs(y - c) ** p), w ** beta)

    def w_funct(D, p):
        def inner(x):
            res = 1 / (np.sum((x / D) ** (1 / (p - 1)), 0))
            if not D.any():
                return 1 / len(D)
            if np.isnan(res):
                return 1
            return res

        return np.vectorize(inner)

    origin = minkowski_center(data, p)
    original_data = np.copy(data)
    labels = np.zeros(shape=len(data), dtype=int)
    centroids = []
    weights = []
    indices = np.arange(len(data))

    V = data.shape[1]  # features count
    cluster_label = 1
    while len(data) > 1:
        w = np.full(shape=(2, V), fill_value=1 / V)
        x_origin = np.apply_along_axis(dist_to_p_beta(origin, p, beta, w[0]), axis=1, arr=data)
        ct_i = np.argmax(x_origin)
        ct = data[ct_i]
        ct_hist = collections.deque([None, None, ct], maxlen=3)
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[ct_i] = True
        while not (np.array_equal(ct_hist[-1], ct_hist[-2]) or np.array_equal(ct_hist[-1], ct_hist[-3])):
            #  w[0] for centroid cluster and w[1] for anomalous cluster
            x_origin = np.apply_along_axis(dist_to_p_beta(origin, p, beta, w[0]), axis=1, arr=data)
            x_ct = np.apply_along_axis(dist_to_p_beta(ct, p, beta, w[1]), axis=1, arr=data)
            anomaly = x_ct < x_origin
            ct = minkowski_center(data[anomaly], p)
            ct_hist.append(ct)
            # update weights
            normalcy = ~anomaly
            norm_data, anom_data = data[normalcy], data[anomaly]
            D0 = np.sum(np.abs(norm_data - origin) ** p, axis=0)
            w[0] = w_funct(D0, p)(D0)
            D1 = np.sum(np.abs(anom_data - ct) ** p, axis=0)
            w[1] = w_funct(D1, p)(D1)
            # plot(original_data, labels)
            # plt.scatter(ct[0], ct[1], s=250, marker='x', color='k')
            # plt.scatter(ct[0], ct[1], s=250, marker='o', facecolors='none', edgecolors='k')
            # plt.scatter(origin[0], origin[1], s=250, marker='x', color='k')
            # plt.scatter(origin[0], origin[1], s=250, marker='o', facecolors='none', edgecolors='k')
        centroids.append(ct)
        weights.append(w[1])
        data = data[normalcy]
        indices = indices[normalcy]
        labels[indices] = cluster_label
        cluster_label += 1
    centroids.append(data[0])
    weights.append(np.full(shape=V, fill_value=1 / V))
    return labels, centroids, weights


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test5.dat")
    l, c, w = a_pattern_init_p_beta(data, p=10, beta=1)
    hold_plot(data, l)
