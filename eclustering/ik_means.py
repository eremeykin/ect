__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
from tests.tools.plot import plot


def ik_means(data):
    original_data = data
    data = np.copy(data)
    indices = np.arange(len(data))
    labels = np.zeros(len(data), dtype=int)
    centroids = []

    origin = np.mean(data, 0)  # TODO check if origin depends of metric

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
    return labels, np.array(centroids)


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test2.dat")
    l, c = ik_means(data)
