import numpy as np

from clustering.agglomerative.algorithms.a_ward import a_ward
from tests.tools import transformation_exists

DATA_DIR = "data/a_ward/"


def test_symmetric_16points():
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    result = a_ward(data, 5, np.arange(0, len(data), dtype=int))
    actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    assert transformation_exists(actual, result)


def test_iris():
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    result = a_ward(data, 3, np.arange(0, len(data), dtype=int))
    # compare with sklearn implementation
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=3)
    model.fit(data)
    actual = model.labels_
    assert transformation_exists(actual, result)
