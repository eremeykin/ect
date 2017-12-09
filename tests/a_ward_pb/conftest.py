import os
from collections import namedtuple

import numpy as np
import pytest

from clustering.agglomerative.utils.matlab_compatible import IMWKMeansClusterStructureMatlabCompatible
from clustering.ik_means.ik_means import IKMeans
from clustering.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from tests.tools import rp, matlab_connector

DataKStarPBRes = namedtuple('DataCSKStarPBRes', 'data k_star p beta res')
DataCSKStarPBRes = namedtuple('DataCSKStarPBRes', 'data cs k_star p beta res')


def short_path(path):
    return path[path.rfind('/') + 1:]


def repr_param(param):
    res = param.res
    sp = short_path(param.data)
    ks = param.k_star
    return "{:10s}, k_star={:4d}, p= {}, beta = {}, {}".format(res, ks, param.p, param.beta, sp)


@pytest.fixture(params=[
    DataKStarPBRes(rp("shared/data/data500ws.pts"), 12, 2, 2, 'matlab'),
    DataKStarPBRes(rp("shared/data/data500ws.pts"), 8, 2, 2, 'matlab'),
    DataKStarPBRes(rp("shared/data/iris.pts"), 3, 2, 2, 'matlab'),
    DataKStarPBRes(rp("shared/data/iris.pts"), 3, 3, 2, 'matlab'),
],
    ids=repr_param)
def data_cs_k_star_res(request):
    param = request.param
    res = dict()
    if param.res == 'matlab':
        data = np.loadtxt(param.data)
        run_ap_init_pb = APInitPBMatlabCompatible(data, param.p, param.beta)
        run_ap_init_pb()
        # change cluster structure to matlab compatible
        clusters = run_ap_init_pb.cluster_structure.clusters
        new_cluster_structure = IMWKMeansClusterStructureMatlabCompatible(data, param.p, param.beta)
        new_cluster_structure.add_all_clusters(clusters)
        run_ik_means = IKMeans(new_cluster_structure)
        run_ik_means()
        cs = run_ik_means.cluster_structure

        data_file = "'" + os.path.abspath(param.data) + "'"
        matlab_result = matlab_connector('test_a_ward_pb', data_file, param.k_star, param.p, param.beta, 1)
        res['labels'] = matlab_result['Labels'].flatten().astype(int)
        return DataCSKStarPBRes(param.data, cs, param.k_star, param.p, param.beta, res)

    raise AssertionError("wrong test params: {}".format(param.res))
