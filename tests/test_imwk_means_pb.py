from clustering.common import minkowski_center
from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.ik_means.imwk_means_pb import IMWKMeansPB
from tests.parameters import DATA_DIR
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from tests.tools import transformation_exists, matlab_connector
from tests.test_ap_init_pb import _TestAPInitPB, _TestAWardPBCluster
import os


class _TestIMWKMeansPB(IMWKMeansPB):
    """Do not use it. Special implementation of APInitPB for testing purposes only."""

    def _new_cluster(self, label, data):
        return _TestAWardPBCluster(label, data, self._p, self._beta)


def test_iris_sklearn():
    p, beta = 2, 0
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_symmetric_15points():
    p, beta = 2, 0
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_500_random():
    p, beta = 2, 0
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_iris_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}iris.pts'.format(DATA_DIR)
    data = np.loadtxt(data_path)
    run_ap_init_pb = _TestAPInitPB(data, p, beta)
    ap_init_pb_result = run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    run_imwk_means = _TestIMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    data_file = "'" + os.path.abspath(data_path) + "'"
    matlab_result = matlab_connector('test_imwk_means_pb', data_file, threshold, p, beta)
    matlab_result = [int(i) for i in matlab_result]
    print("matlab_result = {}".format(matlab_result))
    print("my result =     {}".format(list(imwk_means_result)))
    assert transformation_exists(matlab_result, imwk_means_result)
