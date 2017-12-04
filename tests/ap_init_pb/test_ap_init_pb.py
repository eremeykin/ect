from clustering.pattern_initialization.ap_init_pb import APInitPB
from clustering.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from tests.tools import transformation_exists
import numpy as np
from tests.tools import matlab_connector, array_equals_up_to_order


def test_ap_init_pb(data_pb_res):
    p, beta = data_pb_res.p, data_pb_res.beta
    res = data_pb_res.res
    data = np.loadtxt(data_pb_res.data)
    run_api_pb = APInitPB(data, p=p, beta=beta)
    try:
        if data_pb_res.mlc:
            run_api_pb = APInitPBMatlabCompatible(data, p=p, beta=beta)
    except AttributeError:
        pass
    my_labels = run_api_pb()
    clusters = run_api_pb.cluster_structure.clusters
    my_weights = np.array([c.weights for c in clusters])
    my_centroids = np.array([c.centroid for c in clusters]).astype(float)

    assert transformation_exists(res['labels'], my_labels)
    if 'weights' in res:
        assert array_equals_up_to_order(res['weights'], my_weights, atol=1.e-3)
    if 'centroids' in res:
        assert array_equals_up_to_order(res['centroids'], my_centroids, atol=1.e-3)