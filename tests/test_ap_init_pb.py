from clustering.pattern_initialization.ap_init_pb import APInitPB
from tests.tools import transformation_exists
import collections
import numpy as np
import os
from scipy.spatial.distance import minkowski
from clustering.common import get_weights, minkowski_center, weighed_minkowski
from tests.parameters import DATA_DIR
from tests.tools import matlab_connector
from clustering.agglomerative.agglomerative_cluster import AWardPBCluster


class _TestAWardPBCluster(AWardPBCluster):
    """Do not use it. Special implementation of AWardPBCluster for testing purposes only.

    Ok. If You are still here, I'll told you what happened. When I explored canonical matlab
    implementation of  APInitPB (see inside iMWKmeans), I found to issues:
        * for weights calculations they use 1/(1-beta) power instead of 1/(1-p)
        * they use D = D + 0.01 to avoid zero division. It's not quite accurate.
    So, this version apples this corrections for testing only. I will probably use
    my implementation in general. That is why all '..._matlab' tests use _TestAWardPBCluster version.
    """

    def _update(self):
        """Updates cluster centroid and weights"""
        # centroid update to the component-wise Minkowski centre of all points
        cluster_points = self._data[self._points_indices]
        self._centroid = minkowski_center(cluster_points, self._p)
        # weights update (as per 7)
        D = np.sum(np.abs(cluster_points - self.centroid) ** self._p, axis=0).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            D += 0.01
            denominator = ((D ** (1 / (self._p - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (self._beta - 1))))
        isnan = np.isnan(denominator)
        if np.any(isnan):
            self._weights = isnan.astype(int) / np.sum(isnan)
        else:
            self._weights = np.float64(1.0) / denominator
        self._is_stable = False
        assert self._weights.shape == (self._dim_cols,)
        assert np.abs(np.sum(self._weights) - 1) < 0.0001


class _TestAPInitPB(APInitPB):
    """Do not use it. Special implementation of APInitPB for testing purposes only."""

    def _new_cluster(self, label, data):
        return _TestAWardPBCluster(label, data, self._p, self._beta)


def test_symmetric_15points():
    p, beta = 2, 2
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=beta, beta=beta)
    result = run_api_p_beta()
    actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    naive_result, c, w = _naive_ap_init_pb(data, p=2, beta=2)
    assert transformation_exists(actual, result)
    assert transformation_exists(naive_result, result)


def test_iris():
    p, beta = 2, 2
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    naive_result, c, w = _naive_ap_init_pb(data, p=p, beta=beta)
    assert transformation_exists(naive_result, result)


def test_500_random():
    p, beta = 3, 2
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    naive_result, c, w = _naive_ap_init_pb(data, p=p, beta=beta)
    assert transformation_exists(naive_result, result)


def test_iris_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}iris.pts'.format(DATA_DIR)
    data = np.loadtxt(data_path)
    run_api_p_beta = _TestAPInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    data_file = "'" + os.path.abspath(data_path) + "'"
    matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, p, beta)
    matlab_labels = matlab_result['AnomalousLabels'].flatten().astype(int)
    matlab_weights = matlab_result['InitW']
    matlab_centroids = matlab_result['InitZ']
    my_weights = np.array([c._weights for c in run_api_p_beta.clusters])
    my_centroids = np.array([c.centroid for c in run_api_p_beta.clusters]).astype(float)
    assert np.allclose(my_weights, matlab_weights, atol=1.e-5)
    assert np.allclose(my_centroids, matlab_centroids, atol=1.e-5)
    assert transformation_exists(matlab_labels, result)


def test_symmetric_15points_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}symmetric_15points.pts'.format(DATA_DIR)
    data = np.loadtxt(data_path)
    run_api_p_beta = _TestAPInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    data_file = "'" + os.path.abspath(data_path) + "'"
    matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, p, beta)
    matlab_result = [int(i) for i in matlab_result]
    assert transformation_exists(matlab_result, result)


def test_500_random_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}data500ws.pts'.format(DATA_DIR)
    data = np.loadtxt(data_path)
    run_api_p_beta = _TestAPInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    data_file = "'" + os.path.abspath(data_path) + "'"
    matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, p, beta)
    matlab_result = [int(i) for i in matlab_result]
    assert transformation_exists(matlab_result, result)


def _naive_ap_init_pb(data, p, beta):
    data_copy = np.copy(data)
    centroids = []
    weights = []
    indices = np.arange(len(data))
    labels = np.zeros(len(data), dtype=int)
    V = data.shape[1]

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
            # tobj.plot(data, labels[indices], centroids, show_num=False)
            x_to_origin = np.apply_along_axis(lambda x: weighed_minkowski(x, origin, p, tent_w, beta), axis=1, arr=data)
            x_to_ct = np.apply_along_axis(lambda x: weighed_minkowski(x, ct, p, tent_w, beta), axis=1, arr=data)
            anomaly = x_to_ct < x_to_origin
            # normalcy = ~anomaly
            ct = minkowski_center(data[anomaly], p)
            tent_w = get_weights(data[anomaly], ct, p)
            ct_queue.append(ct)
        normalcy = ~anomaly
        centroids.append(ct)
        # centroids = []
        weights.append(w)
        data = data[normalcy]
        indices = indices[normalcy]
        cluster_label += 1
        labels[indices] = cluster_label
    if len(data) > 0:
        ct = minkowski_center(data, p)
        centroids.append(ct)
        weights.append(np.full(ct.shape, 1 / len(ct)))
    return labels, np.array(centroids), np.array(weights)
