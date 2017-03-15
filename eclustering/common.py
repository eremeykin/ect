from scipy.optimize import fmin_tnc
import numpy as np


def minkowski_center(data, p):
    def D(X, a):
        return np.sum(np.abs(X - a) ** p) / len(X)

    def component_center(X):
        return fmin_tnc(func=lambda x: D(X, x), x0=np.mean(X), approx_grad=True, disp=0)[0]

    if p == 1:
        return np.median(data, 0)
    if p == 2:
        return np.mean(data, 0)
    return np.apply_along_axis(component_center, 0, data)[0]  # TODO invent something more elegant


# see formula (7)
def get_weights(cluster_k, ct_k, p):  # cluster_k is cluster data for k cluster and ct_k is it's centroid
    D = np.sum((np.abs(cluster_k - ct_k)) ** p, axis=0)
    D = np.resize(D, (len(D), len(D)))
    if np.any(D == 0):  # TODO actualy == 0 is not good scheme
        return np.full(ct_k.shape, 1 / len(ct_k.shape))
    D_inv = np.resize(1 / D, (len(D), len(D)))
    M = np.multiply(D, D_inv.T) ** (1 / (p - 1))
    res = 1 / np.sum(M, axis=0)
    return res


# see formula (9)
def weighed_minkowski(x, y, p, w, beta):
    return np.dot(np.abs(x - y) ** p, w ** beta)
