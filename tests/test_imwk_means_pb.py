from clustering.common import minkowski_center
from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.ik_means.imwk_means_pb import IMWKMeansPB
from tests.parameters import DATA_DIR
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from tests.tools import transformation_exists, matlab_connector, array_equals_up_to_order
import os
from clustering.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from clustering.agglomerative.agglomerative_cluster_structure import IMWKMeansClusterStructureMatlabCompatible

def test_iris_sklearn():
    p, beta = 2, 0
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.cluster_structure.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(run_ap_init_pb.cluster_structure)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_symmetric_15points():
    p, beta = 2, 0
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.cluster_structure.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(run_ap_init_pb.cluster_structure)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_500_random():
    p, beta = 2, 0
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.cluster_structure.clusters
    centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(run_ap_init_pb.cluster_structure)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_iris_matlab():
    p, beta = 2, 2
    threshold = 0
    data_path = '{}iris.pts'.format(DATA_DIR)
    matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels = \
        _my_vs_matlab(p, beta, threshold, data_path)
    _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels)


def _my_vs_matlab(p, beta, threshold, data_path):
    # load data
    data = np.loadtxt(data_path)
    data_file = "'" + os.path.abspath(data_path) + "'"
    # run my implementation
    run_ap_init_pb = APInitPBMatlabCompatible(data, p=p, beta=beta)
    init_labels = run_ap_init_pb()
    clusters = run_ap_init_pb.cluster_structure.clusters
    new_cluster_structure = IMWKMeansClusterStructureMatlabCompatible(data, p, beta)
    new_cluster_structure.add_all_clusters(clusters)
    run_imwk_means = IMWKMeansPB(new_cluster_structure)
    my_labels = run_imwk_means()
    result_clusters = run_imwk_means.cluster_structure.clusters
    my_weights = np.array([c.weights for c in result_clusters])
    my_centroids = np.array([c.centroid for c in result_clusters]).astype(float)
    # run matlab implementation
    matlab_result = matlab_connector('test_imwk_means_pb', data_file, threshold, p, beta)
    matlab_labels = matlab_result['Labels'].flatten().astype(int)
    matlab_weights = matlab_result['FinalW']
    matlab_centroids = matlab_result['FinalZ']
    return matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels


def _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels):
    assert transformation_exists(matlab_labels, my_labels)
    assert array_equals_up_to_order(my_centroids, matlab_centroids, atol=1.e-5)
    assert array_equals_up_to_order(my_weights, matlab_weights, atol=1.e-3)

