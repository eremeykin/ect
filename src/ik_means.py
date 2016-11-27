__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
import matplotlib.pyplot as plt


def ik_means(data, theta=1):
    plt.ion()
    print(data)



    n = data.shape[0]
    cy = np.mean(data, 0)
    dist_to = lambda y: lambda x: d.sqeuclidean(x, y)
    x_cy = np.apply_along_axis(dist_to(cy), 1, data)
    anomalies = []
    not_removed = np.full(len(data), True, dtype=bool)
    while len(data) > 1:
        cti = np.argmax(x_cy)
        ct = data[cti]
        ct_old = None
        anomaly = np.full(len(data), False, dtype=bool)
        # anomaly = anomaly_old
        anomaly[cti] = True
        while not np.array_equal(ct_old, ct):  # not np.array_equal(anomaly_old, anomaly):
            print(data)
            plt.clf()
            plt.scatter(data[:, 0], data[:, 1])
            plt.scatter(ct[0], ct[1], marker='*', s=200, color='yellow')
            plt.scatter(cy[0], cy[1], marker='*', s=200, color='blue')
            plt.xlim(0, 30)
            plt.ylim(-4, 4)
            plt.pause(0.05)

            ct_old=np.copy(ct)
            x_ct = np.apply_along_axis(dist_to(ct), 1, data)
            anomaly = x_ct < x_cy
            ct = np.mean(data[np.where(anomaly)], 0)

        data = data[np.where(np.logical_not(anomaly))]
        x_cy = x_cy[np.where(np.logical_not(anomaly))]



data = np.loadtxt('../tests/data/test2.dat')
# data = np.array([[1, 2, 0], [1, 0, 6], [4, 3, 6], [0, 20, 4], [4, 2, 5]])
# data = np.array([[1, 2], [1, 0], [4, 3], [0, 20], [4, 2]])
ik_means(data)