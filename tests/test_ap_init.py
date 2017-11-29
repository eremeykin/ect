import numpy as np

from clustering.pattern_initialization.ap_init import APInit
from tests.tools import transformation_exists
from tests.parameters import DATA_DIR

def test_symmetric_15points():
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_ap_init = APInit(data)
    result = run_ap_init()
    actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    assert transformation_exists(actual, result)


def test_iris():
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_ap_init = APInit(data)
    result = run_ap_init()
    actual, _ = _naive_ap_init(data)
    assert transformation_exists(actual, result)


def test_500_random():
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_ap_init = APInit(data)
    result = run_ap_init()
    naive_result, _ = _naive_ap_init(data)
    assert transformation_exists(naive_result, result)


def _naive_ap_init(data):
    from scipy.spatial.distance import sqeuclidean as se_dist
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
        normalcy = ~anomaly
        centroids.append(ct)
        data = data[normalcy]
        x_to_origin = x_to_origin[normalcy]
        indices = indices[normalcy]
        cluster_label += 1
        labels[indices] = cluster_label
    if len(data) > 0:
        centroids.append(data[0])
    return labels, np.array(centroids)
