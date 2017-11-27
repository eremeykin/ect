import numpy as np

from clustering.pattern_initialization.anomalous_cluster_p_beta import APInitPB


DATA_DIR = "data/"


def test_symmetric_16points():
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    # data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=2, beta=2)
    result = run_api_p_beta()
    print(run_api_p_beta._clusters)
    print(result)
    # actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    # assert transformation_exists(actual, result)

#
# def test_iris():
#     data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
#     result = a_ward(data, 3, np.arange(0, len(data), dtype=int))
#     # compare with sklearn implementation
#     from sklearn.cluster import AgglomerativeClustering
#     model = AgglomerativeClustering(n_clusters=3)
#     model.fit(data)
#     actual = model.labels_
#     assert transformation_exists(actual, result)
