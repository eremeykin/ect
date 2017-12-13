import numpy as np

from clustering.agglomerative.ik_means.ik_means import IKMeans
from tests.tools import transformation_exists, array_equals_up_to_order


def test_imwk_means(data_cs_pb_res):
    run_imwk_means = IKMeans(data_cs_pb_res.cs)
    my_labels = run_imwk_means()
    result_clusters = run_imwk_means.cluster_structure.clusters
    my_weights = np.array([c.weights for c in result_clusters])
    my_centroids = np.array([c.centroid for c in result_clusters]).astype(float)
    assert transformation_exists(data_cs_pb_res.res['labels'], my_labels)
    if 'centroids' in data_cs_pb_res.res:
        assert array_equals_up_to_order(data_cs_pb_res.res['centroids'], my_centroids, atol=1.e-3)
    if 'weights' in data_cs_pb_res.res:
        assert array_equals_up_to_order(data_cs_pb_res.res['weights'], my_weights, atol=1.e-3)
