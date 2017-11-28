import numpy as np

from clustering.pattern_initialization.ap_init_pb import APInitPB
from tests.tools import transformation_exists
from tests.ap_init_pb.naive_ap_init_pb import anomalous_cluster_p_beta

DATA_DIR = "/home/eremeykin/d_disk/projects/Clustering/ect/tests/shared/data/"


def test_symmetric_16points():
    p, beta = 2, 2
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=beta, beta=beta)
    result = run_api_p_beta()
    actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    naive_result, c, w = anomalous_cluster_p_beta(data, p=2, beta=2)
    assert transformation_exists(actual, result)
    assert transformation_exists(naive_result, result)


def test_iris():
    p, beta = 2, 2
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    naive_result, c, w = anomalous_cluster_p_beta(data, p=p, beta=beta)
    assert transformation_exists(naive_result, result)


def test_500_random():
    p, beta = 3, 2
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=p, beta=beta)
    result = run_api_p_beta()
    naive_result, c, w = anomalous_cluster_p_beta(data, p=p, beta=beta)
    assert transformation_exists(naive_result, result)
