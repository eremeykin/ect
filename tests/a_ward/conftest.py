import os
from collections import namedtuple

import numpy as np
import pytest
from clustering.agglomerative.ik_means.ik_means import IKMeans
from sklearn.cluster import AgglomerativeClustering as sklearn_clustering

from clustering.agglomerative.pattern_initialization.ap_init import APInit
from clustering.agglomerative.utils.a_ward_cluster_structure import AWardClusterStructure
from tests.tools import rp, matlab_connector

DataKStarRes = namedtuple('DataKStarRes', 'data k_star res')
DataCSKStarRes = namedtuple('DataKStarRes', 'data cs k_star res')


def short_path(path):
    return path[path.rfind('/') + 1:]


def repr_param(param):
    res = param.res
    sp = short_path(param.data)
    ks = param.k_star
    return "{:10s}, k_star={:4d}, {}".format(res, ks, sp)


@pytest.fixture(params=[
    DataKStarRes(rp("shared/data/symmetric_15points.pts"), 5, 'labels'),
    DataKStarRes(rp("shared/data/symmetric_15points.pts"), 3, 'sklearn'),
    DataKStarRes(rp("shared/data/iris.pts"), 3, 'sklearn'),
    DataKStarRes(rp("shared/data/data500ws.pts"), 12, 'sklearn'),
    DataKStarRes(rp("shared/data/data500ws.pts"), 12, 'matlab'),
    DataKStarRes(rp("shared/data/data500ws.pts"), 8, 'matlab'),
    DataKStarRes(rp("shared/data/iris.pts"), 3, 'matlab'),
],
    ids=repr_param)
def data_cs_k_star_res(request):
    param = request.param
    if param.res == 'labels':
        data = np.loadtxt(param.data)
        labels = np.loadtxt(param.data[:param.data.rfind('.')] + ".lbs")
        cluster_structure = AWardClusterStructure.from_labels(data, labels)
        return DataCSKStarRes(param.data, cluster_structure, param.k_star, labels)
    if param.res == 'sklearn':
        data = np.loadtxt(param.data)
        model = sklearn_clustering(n_clusters=param.k_star)
        model.fit(data)
        cluster_structure = AWardClusterStructure.from_labels(data, np.arange(0, len(data), dtype=int))
        return DataCSKStarRes(param.data, cluster_structure, param.k_star, model.labels_)
    if param.res == 'matlab':
        data = np.loadtxt(param.data)
        run_ap_init = APInit(data)
        run_ap_init()
        run_ik_means = IKMeans(run_ap_init.cluster_structure)
        run_ik_means()
        cs = run_ik_means.cluster_structure

        data_file = "'" + os.path.abspath(param.data) + "'"
        # -                                                                         p  b init
        matlab_result = matlab_connector('test_a_ward_pb', data_file, param.k_star, 2, 0, 1)
        res = matlab_result['Labels'].flatten().astype(int)

        return DataCSKStarRes(param.data, cs, param.k_star, res)

    raise AssertionError("wrong test params: {}".format(param.res))
