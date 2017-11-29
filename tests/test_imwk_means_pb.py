from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.ik_means.imwk_means_pb import IMWKMeansPB
from tests.parameters import DATA_DIR
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from tests.tools import transformation_exists


def test_iris():
    p, beta = 2, 0
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    clusters_num = len(clusters)
    cluster_centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=clusters_num, n_init=1, init=cluster_centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_symmetric_15points():
    p, beta = 2, 0
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    cluster_centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=cluster_centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)


def test_500_random():
    p, beta = 2, 0
    data = np.loadtxt('{}data500ws.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    run_ap_init_pb()
    clusters = run_ap_init_pb.clusters
    cluster_centroids = np.array([cluster.centroid for cluster in clusters])
    k_means = sklearnKMeans(n_clusters=len(clusters), n_init=1, init=cluster_centroids, algorithm='full').fit(data)
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    assert transformation_exists(k_means.labels_, imwk_means_result)
