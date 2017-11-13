import numpy as np
from clustering.agglomerative.a_ward import a_ward
import time
DATA_DIR = "data/a_ward/"


def test_symmetric_16points():
    data = np.loadtxt('{}symmetric_16points.pts'.format(DATA_DIR))
    result = a_ward(data, 5, np.arange(0, len(data), dtype=int))
    labels = np.loadtxt('{}symmetric_16points.lbs'.format(DATA_DIR), dtype=int)
    assert np.array_equal(labels, result)
#
# def test_time():
#     files = ['data10bs','data10ms','data10ws',
#                  'data100bs', 'data100ms', 'data100ws',
#                  'data1000bs', 'data1000ms', 'data1000ws',
#                  'data10000bs', 'data10000ms', 'data1000ws']
#     for file in files:
#         start = time.time()
#         print({})
#