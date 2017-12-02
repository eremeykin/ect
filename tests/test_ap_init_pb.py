from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from tests.tools import transformation_exists
import collections
import numpy as np
import os
from scipy.spatial.distance import minkowski
from clustering.common import get_weights, minkowski_center, weighed_minkowski
from tests.parameters import DATA_DIR
from tests.tools import matlab_connector, array_equals_up_to_order


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
    matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels = \
        _my_vs_matlab(p, beta, threshold, data_path)
    _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels)


def test_symmetric_15points_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}symmetric_15points.pts'.format(DATA_DIR)
    matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels = \
        _my_vs_matlab(p, beta, threshold, data_path)
    matlab_weights[-1, :] = np.array([0.5, 0.5])  # replace Nan from matlab
    matlab_centroids[-1, :] = np.array([0, 0])  # replace Nan from matlab
    _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels)


def test_500_random_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}data500ws.pts'.format(DATA_DIR)
    matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels = \
        _my_vs_matlab(p, beta, threshold, data_path)
    _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels)


def _my_vs_matlab(p, beta, threshold, data_path):
    data = np.loadtxt(data_path)
    data_file = "'" + os.path.abspath(data_path) + "'"
    run_api_pb = APInitPBMatlabCompatible(data, p=p, beta=beta)
    my_labels = run_api_pb()
    clusters = run_api_pb.cluster_structure.clusters
    my_weights = np.array([c.weights for c in clusters])
    my_centroids = np.array([c.centroid for c in clusters]).astype(float)
    # run matlab implementation
    matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, p, beta)
    matlab_labels = matlab_result['AnomalousLabels'].flatten().astype(int)
    matlab_weights = matlab_result['InitW']
    matlab_centroids = matlab_result['InitZ']

    return matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels


def _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels):
    assert array_equals_up_to_order(my_weights, matlab_weights, atol=1.e-5)
    assert array_equals_up_to_order(my_centroids, matlab_centroids, atol=1.e-5)
    assert transformation_exists(matlab_labels, my_labels)


def _naive_ap_init_pb(data, p, beta):
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
            x_to_origin = np.apply_along_axis(lambda x: weighed_minkowski(x, origin, p, tent_w, beta), axis=1,
                                              arr=data)
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
