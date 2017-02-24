from tests.tools.plot import TestObject

__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
# from tests.tools.plot import plot, hold_plot
from eclustering.common import minkowski_center, weights_function
import collections

import matplotlib

matplotlib.use('Qt5Agg')


def a_pattern_init(data, tobj=None):
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
        tobj.plot(data, labels[indices], prefix='test')
        normalcy = ~anomaly
        centroids.append(ct)
        data = data[normalcy]
        x_origin = x_origin[normalcy]
        indices = indices[normalcy]
        labels[indices] = cluster_label
        cluster_label += 1
    if len(data) > 0:
        centroids.append(data[0])
    tobj.plot(original_data, labels, prefix='test')
    return labels, np.array(centroids)


def a_pattern_init_p_beta(data, p, beta, tobj=None):
    def dist_to_p_beta(c, w):
        return lambda y: np.dot((np.abs(y - c) ** p), w ** beta)

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
        x_origin = np.apply_along_axis(dist_to_p_beta(origin, w[0]), axis=1, arr=data)
        ct_i = np.argmax(x_origin)
        ct = data[ct_i]
        print('ct = '+str(ct))
        ct_hist = collections.deque([None, None, ct], maxlen=3)
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[ct_i] = True
        while not (np.array_equal(ct_hist[-1], ct_hist[-2]) or np.array_equal(ct_hist[-1], ct_hist[-3])):
            print('____before____')
            print('ct='+str(ct))
            print('W=')
            print(w)
            #  w[0] for centroid cluster and w[1] for anomalous cluster
            x_origin = np.apply_along_axis(dist_to_p_beta(origin, w[0]), axis=1, arr=data)
            x_ct = np.apply_along_axis(dist_to_p_beta(ct, w[1]), axis=1, arr=data)
            anomaly = x_ct < x_origin
            ct = minkowski_center(data[anomaly], p)
            ct_hist.append(ct)
            # update weights
            normalcy = ~anomaly
            norm_data, anom_data = data[normalcy], data[anomaly]
            D0 = np.sum(np.abs(norm_data - origin) ** p, axis=0)
            w[0] = weights_function(D0, D=D0, p=p)
            D1 = np.sum(np.abs(anom_data - ct) ** p, axis=0)
            w[1] = weights_function(D1, D=D1, p=p)
            print('____after____')
            print('ct='+str(ct))
            print('W=')
            print(w)
        print('centroids='+str(centroids))
        print('___________cut____________')
        tobj.plot(data, labels[indices], centroids, prefix='a_pattern_init_p_beta')
        centroids.append(ct)
        weights.append(w[1])
        data = data[normalcy]
        indices = indices[normalcy]
        labels[indices] = cluster_label
        cluster_label += 1
    tobj.plot(original_data, labels, prefix='a_pattern_init_p_beta')
    if len(data) > 0:
        centroids.append(data[0])
        weights.append(np.full(shape=V, fill_value=1 / V))
    return labels, np.array(centroids), np.array(weights)


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test8.dat")
    # a_pattern_init(data, tobj=TestObject('test'))
    # exit()
    p, beta = 2, 2
    labels, centroids = a_pattern_init(data, tobj=TestObject(test_name='ikmeans_test2.dat'))

    labels, centroids, weights = a_pattern_init_p_beta(data, p=p, beta=beta,
                                                       tobj=TestObject(test_name='ikmeans_test2.dat'))
