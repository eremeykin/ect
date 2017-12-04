import pytest
from collections import namedtuple
import numpy as np
import os
from tests.tools import matlab_connector, rp

DataRes = namedtuple('DataRes', 'data res')


def short_path(path):
    return path[path.rfind('/') + 1:]


def repr_param(param):
    res = param.res
    sp = short_path(param.data)
    return "{:10s}, {}".format(res, sp)


@pytest.fixture(
    params=[
        DataRes(rp("shared/data/symmetric_15points.pts"), 'labels'),
        DataRes(rp("shared/data/symmetric_15points.pts"), 'naive'),
        DataRes(rp("shared/data/iris.pts"), 'naive'),
        DataRes(rp("shared/data/data500ws.pts"), 'naive'),
        DataRes(rp("shared/data/iris.pts"), 'matlab'),
        DataRes(rp("shared/data/symmetric_15points.pts"), 'matlab'),
        DataRes(rp("shared/data/data500ws.pts"), 'matlab'),
    ],
    ids=repr_param)
def data_res(request):
    param = request.param
    res = dict()
    if param.res == 'labels':
        res['labels'] = np.loadtxt(param.data[:param.data.rfind('.')] + ".lbs")
        return DataRes(param.data, res)
    if param.res == 'naive':
        data = np.loadtxt(param.data)
        naive_result, c = _naive_ap_init(data)
        res['labels'] = naive_result
        res['centroids'] = c
        return DataRes(param.data, res)
    if param.res == "matlab":
        data_file = "'" + os.path.abspath(param.data) + "'"
        # run matlab implementation
        threshold = 0
        matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, 2, 0)
        res['labels'] = matlab_result['AnomalousLabels'].flatten().astype(int)
        # replace nans in centroids
        centroids = matlab_result['InitZ']
        centroids[np.isnan(centroids)] = 0  # TODO is it correct?
        res['centroids'] = centroids
        return DataRes(param.data, res)

    raise AssertionError("wrong test params: {}".format(param.res))


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
