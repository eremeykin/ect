from clustering.common import minkowski_center
from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.ik_means.imwk_means_pb import IMWKMeansPB
from tests.parameters import DATA_DIR
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from tests.tools import transformation_exists, matlab_connector, array_equals_up_to_order
from tests.test_ap_init_pb import _TestAPInitPB
from clustering.agglomerative.agglomerative_cluster import AWardPBCluster
import os


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
    matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels = \
        _my_vs_matlab(p, beta, threshold, data_path)
    _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels)


def _my_vs_matlab(p, beta, threshold, data_path):
    # load data
    data = np.loadtxt(data_path)
    data_file = "'" + os.path.abspath(data_path) + "'"
    # run my implementation
    run_ap_init_pb = _TestAPInitPB(data, p=p, beta=beta)
    init_labels = run_ap_init_pb()
    clusters = run_ap_init_pb.clusters

    def get_mean_D():
        D_list = []
        for cluster in clusters:
            # think twice! Cluser may not be updated yet
            cluster_points = cluster._data[cluster._points_indices]
            cluster_centroid = minkowski_center(cluster_points, cluster._p)
            cluster_D = np.sum(np.abs(cluster_points - cluster_centroid) ** cluster._p, axis=0).astype(np.float64)
            D_list.append(cluster_D)
        D_array = np.array(D_list)
        return np.mean(D_array)

    class Mock:

        def __init__(self, cluster):
            self.clst = cluster

        def update(self):
            cluster_points = self.clst._data[self.clst._points_indices]
            self.clst._centroid = minkowski_center(cluster_points, self.clst._p)
            # weights update (as per 7)
            D = np.sum(np.abs(cluster_points - self.clst.centroid) ** self.clst._p, axis=0).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                D += get_mean_D()
                denominator = ((D ** (1 / (self.clst._beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (self.clst._beta - 1))))
            isnan = np.isnan(denominator)
            if np.any(isnan):
                self.clst._weights = isnan.astype(int) / np.sum(isnan)
            else:
                self.clst._weights = np.float64(1.0) / denominator
            self.clst._is_stable = False
            assert self.clst._weights.shape == (self.clst._dim_cols,)
            assert np.abs(np.sum(self.clst._weights) - 1) < 0.0001

    for cluster in clusters:
        cluster._update = Mock(cluster).update

    def new_distance_formula(point1, point2, weights, p, beta):
        return np.sum((np.abs(point1 - point2) ** p) * (weights ** beta)) ** (1/p)

    AWardPBCluster.distance_formula = new_distance_formula


    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    my_labels = run_imwk_means()

    clusters[1]._update()

    my_weights = np.array([c._weights for c in run_imwk_means.clusters])
    my_centroids = np.array([c.centroid for c in run_imwk_means.clusters]).astype(float)
    # run matlab implementation
    matlab_result = matlab_connector('test_imwk_means_pb', data_file, threshold, p, beta)
    matlab_labels = matlab_result['Labels'].flatten().astype(int)
    matlab_weights = matlab_result['FinalW']
    matlab_centroids = matlab_result['FinalZ']
    return matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels


def _assert(matlab_weights, my_weights, matlab_centroids, my_centroids, matlab_labels, my_labels):
    assert transformation_exists(matlab_labels, my_labels)
    assert array_equals_up_to_order(my_centroids, matlab_centroids, atol=1.e-5)
    print("centroids")
    print(my_centroids)
    print(matlab_centroids)
    print()
    print("weights")
    print(my_weights)
    print(matlab_weights)
    print()
    assert array_equals_up_to_order(my_weights, matlab_weights, atol=1.e-4)

