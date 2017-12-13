import os
from collections import namedtuple

import numpy as np
import pytest
from sklearn.cluster import KMeans as sklearnKMeans

from clustering.agglomerative.pattern_initialization.ap_init_pb import APInit
from tests.tools import matlab_connector, rp

DataCSRes = namedtuple('DataCSRes', 'data cs res')
DataRes = namedtuple('DataCSRes', 'data res')


def short_path(path):
    return path[path.rfind('/') + 1:]


def repr_param(param):
    res = param.res
    sp = short_path(param.data)
    return "{:10s},  {}".format(res, sp)


@pytest.fixture(
    params=[
        DataRes(rp("shared/data/symmetric_15points.pts"), 'sklearn'),
        DataRes(rp("shared/data/iris.pts"), 'sklearn'),
        DataRes(rp("shared/data/data500ws.pts"), 'sklearn'),
        DataRes(rp("shared/data/iris.pts"), 'matlab'),
        DataRes(rp("shared/data/data500ws.pts"), 'matlab'),
    ],
    ids=repr_param)
def data_cs_res(request):
    param = request.param
    res = dict()
    if param.res == 'sklearn':
        data = np.loadtxt(param.data)
        run_ap_init = APInit(data)
        run_ap_init()
        clusters = run_ap_init.cluster_structure.clusters
        centroids = np.array([cluster.centroid for cluster in clusters])
        k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
        res['labels'] = k_means.labels_
        return DataCSRes(param.data, run_ap_init.cluster_structure, res)
    if param.res == "matlab":
        data = np.loadtxt(param.data)
        # run my implementation of ap_init
        run_ap_init = APInit(data)
        run_ap_init()
        # run matlab implementation
        data_file = "'" + os.path.abspath(param.data) + "'"
        threshold = 0
        matlab_result = matlab_connector('test_imwk_means_pb', data_file, threshold, 2, 0)
        res['labels'] = matlab_result['Labels'].flatten().astype(int)
        return DataCSRes(param.data, run_ap_init.cluster_structure, res)

    raise AssertionError("wrong test params: {}".format(param.res))
