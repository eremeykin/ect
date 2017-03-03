import numpy as np

from tests.tools.plot import TestObject
from scipy.spatial.distance import sqeuclidean as se_dist


def anomalous_cluster(data, tobj=None):
    data_copy = np.copy(data)
    centroids = []
    indices = np.arange(len(data))
    labels = np.zeros(len(data), dtype=int)

    origin = np.mean(data, axis=0)
    x_to_origin = np.apply_along_axis(lambda x: se_dist(x, origin), axis=1, arr=data)

    cluster_label = 0
    while len(data) > 1:
        ct_index = np.argmax(x_to_origin)
        ct = data[ct_index]
        ct_old = None
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[ct_index] = True
        while not np.array_equal(ct_old, ct):
            ct_old = np.copy(ct)
            x_to_ct = np.apply_along_axis(lambda x: se_dist(x, ct), axis=1, arr=data)
            anomaly = x_to_ct < x_to_origin
            ct = np.mean(data[anomaly], 0)
        tobj.plot(data, labels[[indices]], prefix=str(cluster_label))
        normalcy = ~anomaly
        centroids.append(ct)
        data = data[normalcy]
        x_to_origin = x_to_origin[normalcy]
        indices = indices[normalcy]
        cluster_label += 1
        labels[indices] = cluster_label
    if len(data) > 0:
        centroids.append(data[0])
    tobj.plot(data_copy, labels, prefix="RES")
    return labels, np.array(centroids)

if __name__ == "__main__":
    data = np.loadtxt("../../tests/data/ikmeans_test8.dat")
    tobj = TestObject('anomalous_cluster')
    labels, centroids = anomalous_cluster(data, tobj=tobj)
