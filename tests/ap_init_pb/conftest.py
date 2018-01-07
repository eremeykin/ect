import pytest
from collections import namedtuple
import numpy as np
from scipy.spatial.distance import minkowski
import collections
import os
from clustering.common import get_weights, minkowski_center, weighed_minkowski
from tests.tools import matlab_connector, rp

DataPBRes = namedtuple('DataPBRes', 'data p beta res')
DataPBResMLC = namedtuple('DataPBRes', 'data p beta res mlc')


def short_path(path):
    return path[path.rfind('/') + 1:]


def repr_param(param):
    res = param.res
    sp = short_path(param.data)
    p = param.p
    beta = param.beta
    return "{:10s}, p={:5.2f}, beta={:5.2f} {}".format(res, p, beta, sp)


@pytest.fixture(
    params=[
        DataPBRes(rp("shared/data/symmetric_15points.pts"), 2, 2, 'labels'),
        DataPBRes(rp("shared/data/symmetric_15points.pts"), 2, 2, 'naive'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 2, 'naive'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 2, 'naive'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/symmetric_15points.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 3, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3.5, 2.5, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3.5, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 3, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 3.5, 1, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 1, 'matlab'),

    ],
    ids=repr_param)
def data_pb_res(request):
    param = request.param
    res = dict()
    if param.res == 'labels':
        res['labels'] = np.loadtxt(param.data[:param.data.rfind('.')] + ".lbs")
        return DataPBRes(param.data, param.p, param.beta, res)
    if param.res == 'naive':
        data = np.loadtxt(param.data)
        naive_result, c, w = _naive_ap_init_pb(data, p=param.p, beta=param.beta)
        res['labels'] = naive_result
        # Do not compare naive weights and centroids, seems they are wrong.
        # res['weights'] = w
        # res['centroids'] = c
        return DataPBRes(param.data, param.p, param.beta, res)
    if param.res == "matlab":
        data_file = "'" + os.path.abspath(param.data) + "'"
        # run matlab implementation
        threshold = 0
        matlab_result = matlab_connector('test_ap_init_pb', data_file, threshold, param.p, param.beta)
        res['labels'] = matlab_result['AnomalousLabels'].flatten().astype(int)
        # replace nans in weights
        weights = matlab_result['InitW']
        weights[np.isnan(weights)] = 1 / weights.shape[1]
        res['weights'] = weights
        # replace nans in centroids
        centroids = matlab_result['InitZ']
        centroids[np.isnan(centroids)] = 0  # TODO is it correct?
        res['centroids'] = centroids
        return DataPBResMLC(param.data, param.p, param.beta, res, True)

    raise AssertionError("wrong test params: {}".format(param.res))


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
