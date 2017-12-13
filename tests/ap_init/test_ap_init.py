import numpy as np

from clustering.agglomerative.pattern_initialization.ap_init import APInit
from tests.tools import transformation_exists, array_equals_up_to_order


def test_ap_init(data_res):
    data = np.loadtxt(data_res.data)
    run_ap_init = APInit(data)
    result = run_ap_init()
    clusters = run_ap_init.cluster_structure.clusters
    my_centroids = np.array([c.centroid for c in clusters]).astype(float)
    actual = np.loadtxt(data_res.res['labels'])
    assert transformation_exists(actual, result)
    if 'centroids' in data_res.res:
        assert array_equals_up_to_order(data_res.res['centroids'], my_centroids)
