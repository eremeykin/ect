__author__ = 'eremeykin'
import numpy as np
from scipy.spatial import distance as d
import matplotlib.pyplot as plt


def ik_means(data, theta=1):
    data = np.copy(data)
    indices = np.arange(len(data))
    cy = np.mean(data, 0)
    dist_to = lambda y: lambda x: d.sqeuclidean(x, y)
    x_cy = np.apply_along_axis(dist_to(cy), 1, data)
    labels = np.full(len(data), 0, dtype=int)
    c=1
    while len(data) > 1:
        cti = np.argmax(x_cy)
        ct = data[cti]
        ct_old = None
        anomaly = np.full(len(data), False, dtype=bool)
        anomaly[cti] = True
        while not np.array_equal(ct_old, ct):
            # plt.clf()
            # plt.scatter(data[:, 0], data[:, 1])
            # plt.scatter(ct[0], ct[1], marker='*', s=200, color='yellow')
            # plt.scatter(cy[0], cy[1], marker='*', s=200, color='blue')
            # plt.xlim(0, 100)
            # plt.ylim(0, 100)
            # plt.pause(0.01)
            ct_old=np.copy(ct)
            x_ct = np.apply_along_axis(dist_to(ct), 1, data)
            anomaly = x_ct < x_cy
            ct = np.mean(data[np.where(anomaly)], 0)
        normalcy = np.where(~anomaly)
        data = data[normalcy]
        x_cy = x_cy[normalcy]
        indices = indices[normalcy]
        labels[indices]=c
        c+=1
    # np.savetxt('verify_data/ikmeans_test.dat',labels,fmt="%d")
    return labels