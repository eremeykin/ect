import pytest
from collections import namedtuple
import numpy as np
import os
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
        DataPBRes(rp("shared/data/symmetric_15points.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 3, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3.5, 2.5, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3.5, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3.5, 1, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 1, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 3, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 3.5, 1, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 1, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2.6, 1.1, 'matlab'),
        DataPBRes(rp("shared/data/random_1000x12_c12.pts"), 2.6, 1.1, 'matlab'),

    ],
    ids=repr_param)
def data_pb_res(request):
    param = request.param
    res = dict()
    if param.res == 'labels':
        res['labels'] = np.loadtxt(param.data[:param.data.rfind('.')] + ".lbs")
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
