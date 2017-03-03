import numpy as np
from tests.tools.plot import TestObject
from scipy.spatial.distance import minkowski
import collections
from eclustering.common import minkowski_center


# see formula (7)
def get_weights(cluster_k, ct_k, p):  # cluster_k is cluster data for k cluster and ct_k is it's centroid
    D = np.sum((np.abs(cluster_k - ct_k)) ** p, axis=0)
    D = np.resize(D, (len(D), len(D)))
    D_inv = np.resize(1 / D, (len(D), len(D)))
    M = np.multiply(D, D_inv.T) ** (1 / (p - 1))
    res = 1 / np.sum(M, axis=0)
    return res


# see formula (9)
def weighed_minkowski(x, y, p, w, beta):
    return np.dot(np.abs(x - y) ** p, w ** beta)


def anomalous_cluster_p_beta2(data, p, beta, tobj=None):
    data_copy = np.copy(data)
    centroids = []
    weights = []
    indices = np.arange(len(data))
    labels = np.zeros(len(data), dtype=int)
    V = data.shape[1]

    # TODO define minkowski_center
    origin = minkowski_center(data, p)

    cluster_label = 0
    while len(data) > 1:
        x_to_origin = np.apply_along_axis(lambda x: minkowski(x, origin, p), axis=1, arr=data)
        w = np.full(fill_value=1 / V, shape=V)
        ct_index = np.argmax(x_to_origin)
        ct = data[ct_index]
        ct_queue = collections.deque([None, None, ct], maxlen=3)
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[ct_index] = True
        tent_w = np.full(fill_value=1 / V, shape=V)
        while not (np.array_equal(ct_queue[-1], ct_queue[-2]) or np.array_equal(ct_queue[-1], ct_queue[-3])):
            tobj.plot(data, labels[indices], centroids, show_num=False)
            x_to_origin = np.apply_along_axis(lambda x: weighed_minkowski(x, origin, p, tent_w, beta), axis=1, arr=data)
            x_to_ct = np.apply_along_axis(lambda x: weighed_minkowski(x, ct, p, tent_w, beta), axis=1, arr=data)
            anomaly = x_to_ct < x_to_origin
            # normalcy = ~anomaly
            ct = minkowski_center(data[anomaly], p)
            tent_w = get_weights(data[anomaly], ct, p)
            ct_queue.append(ct)
        normalcy = ~anomaly
        centroids.append(ct)
        centroids = []
        weights.append(w)
        data = data[normalcy]
        indices = indices[normalcy]
        cluster_label += 1
        labels[indices] = cluster_label
        tobj.plot(data, labels[indices], show_num=False)
    return labels, np.array(centroids), np.array(weights)


if __name__ == "__main__":
    data = TestObject.load_data("ikmeans_test8.dat")
    tobj = TestObject('anomalous_cluster_p_beta')
    p, beta = 2, 2
    labels, centroids, weights = anomalous_cluster_p_beta2(data, p, beta, tobj=tobj)
    tobj.plot(data, labels, prefix="RESULT")
