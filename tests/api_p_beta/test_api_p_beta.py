import numpy as np

from clustering.pattern_initialization.anomalous_cluster_p_beta import APInitPB
from tests.tools import transformation_exists


import os

this_file = __file__
i = this_file.rfind(os.path.sep)
DATA_DIR = this_file[:i] + "/data/"
print(this_file)


def test_symmetric_16points():
    data = np.loadtxt('{}symmetric_15points.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=5, beta=5)
    result = run_api_p_beta()
    actual = np.loadtxt('{}symmetric_15points.lbs'.format(DATA_DIR), dtype=int)
    print(result)
    assert transformation_exists(actual, result)


def test_iris():
    data = np.loadtxt('{}iris.pts'.format(DATA_DIR))
    run_api_p_beta = APInitPB(data, p=4, beta=2)
    result = run_api_p_beta()
    print(result)
