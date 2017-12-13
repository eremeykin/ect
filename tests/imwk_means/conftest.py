import os
from collections import namedtuple

import numpy as np
import pytest
from clustering.agglomerative.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from sklearn.cluster import KMeans as sklearnKMeans

from clustering.agglomerative.pattern_initialization.ap_init_pb import APInitPB
from clustering.agglomerative.utils.matlab_compatible import IMWKMeansClusterStructureMatlabCompatible
from tests.tools import matlab_connector, rp

DataCSPBRes = namedtuple('DataCSPBRes', 'data cs p beta res')
DataPBRes = namedtuple('DataCSPBRes', 'data p beta res')


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
        DataPBRes(rp("shared/data/symmetric_15points.pts"), 2, 0, 'sklearn'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 0, 'sklearn'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 0, 'sklearn'),
        DataPBRes(rp("shared/data/iris.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/iris.pts"), 3, 2.5, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 2, 3, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 2, 'matlab'),
        DataPBRes(rp("shared/data/data500ws.pts"), 3, 3, 'matlab'),
    ],
    ids=repr_param)
def data_cs_pb_res(request):
    param = request.param
    res = dict()
    if param.res == 'sklearn':
        if param.p != 2 or param.beta != 0:
            raise AssertionError("Can't test with sklearn when p!=2 or beta!=0. Think about it.")
        data = np.loadtxt(param.data)
        run_ap_init_pb = APInitPB(data, param.p, param.beta)
        run_ap_init_pb()
        clusters = run_ap_init_pb.cluster_structure.clusters
        centroids = np.array([cluster.centroid for cluster in clusters])
        k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
        res['labels'] = k_means.labels_
        return DataCSPBRes(param.data, run_ap_init_pb.cluster_structure, param.p, param.beta, res)
    if param.res == "matlab":
        data = np.loadtxt(param.data)

        # run my implementation of ap_init_pb to setup imwk_means
        run_ap_init_pb = APInitPBMatlabCompatible(data, p=param.p, beta=param.beta)
        init_labels = run_ap_init_pb()
        clusters = run_ap_init_pb.cluster_structure.clusters
        new_cluster_structure = IMWKMeansClusterStructureMatlabCompatible(data, param.p, param.beta)
        new_cluster_structure.add_all_clusters(clusters)

        # run matlab implementation
        data_file = "'" + os.path.abspath(param.data) + "'"
        threshold = 0
        matlab_result = matlab_connector('test_imwk_means_pb', data_file, threshold, param.p, param.beta)
        res['labels'] = matlab_result['Labels'].flatten().astype(int)
        # replace nans in weights
        weights = matlab_result['FinalW']
        weights[np.isnan(weights)] = 1 / weights.shape[1]
        res['weights'] = weights
        # replace nans in centroids
        centroids = matlab_result['FinalZ']
        centroids[np.isnan(centroids)] = 0  # TODO is it correct?
        res['centroids'] = centroids
        return DataCSPBRes(param.data, new_cluster_structure, param.p, param.beta, res)

    raise AssertionError("wrong test params: {}".format(param.res))
