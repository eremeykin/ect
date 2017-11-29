from clustering.common import minkowski_center
from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.ik_means.imwk_means_pb import IMWKMeansPB
from tests.parameters import DATA_DIR
import numpy as np
from sklearn.cluster import KMeans as sklearnKMeans
from tests.tools import transformation_exists

# ref_array = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 3, 7, 3, 4, 3, 7, 4, 5, 4, 3, 7, 3, 3, 7, 5, 4,
#               6, 5, 6, 3, 5, 3, 3, 3, 3, 4, 4, 4, 7, 6, 3, 3, 3, 5, 7, 7, 7, 3, 7, 4, 7, 7, 7, 5, 4, 7, 1, 6, 1, 1, 1,
#               1, 3, 1, 1, 1, 1, 6, 1, 6, 6, 1, 1, 1, 1, 6, 1, 6, 1, 6, 1, 1, 6, 6, 1, 1, 1, 1, 1, 6, 6, 1, 1, 1, 6, 1,
#               1, 1, 6, 1, 1, 1, 6, 1, 1, 6]


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
    ref_array = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 7, 7, 6, 7, 3, 7, 4, 7, 3, 4, 3, 6, 7, 8, 7, 3, 6,
                 7, 6, 5, 6, 7, 7, 7, 7, 7, 5, 7, 6, 6, 6, 6, 5, 3, 7, 7, 7, 3, 6, 3, 7, 6, 4, 3, 3, 3, 7, 4, 3, 1, 5,
                 1, 5, 1, 1, 3, 1, 1, 1, 5, 5, 1, 5, 5, 5, 5, 1, 1, 5, 1, 5, 1, 5, 1, 1, 5, 5, 5, 1, 1, 1, 5, 5, 5, 1,
                 5, 5, 5, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 5]
    assert transformation_exists(ref_array, imwk_means_result)


def test_iris_matlab():
    p, beta = 2, 2
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_ap_init_pb = APInitPB(data, p, beta)
    ap_init_pb_result = run_ap_init_pb()
    print(ap_init_pb_result)
    clusters = run_ap_init_pb.clusters
    run_imwk_means = IMWKMeansPB(clusters, p, beta)
    imwk_means_result = run_imwk_means()
    # for el in imwk_means_result:
    #     print(el)
    ref_array2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,3,7,3,4,3,7,4,5,4,3,7,3,3,7,5,4,6,5,6,3,5,3,3,3,3,4,4,4,7,6,3,3,3,5,7,7,7,3,7,4,7,7,7,5,4,7,1,6,1,1,1,1,3,1,1,1,1,6,1,6,6,1,1,1,1,6,1,6,1,6,1,1,6,6,1,1,1,1,1,6,6,1,1,1,6,1,1,1,6,1,1,1,6,1,1,6]
    from pprint import pprint
    pprint([cluster.centroid for cluster in clusters])
    # print(np.unique(ref_array))
    # print(np.unique(ref_array2))
    # print(np.unique(imwk_means_result))
    assert transformation_exists(ref_array2, imwk_means_result)


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
