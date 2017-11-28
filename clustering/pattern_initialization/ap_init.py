import numpy as np
from scipy.spatial.distance import sqeuclidean as se_dist
from clustering.agglomerative.agglomerative_cluster import AWardCluster
from clustering.pattern_initialization.abstr_ap_init import AbstractAPInit


class APInit(AbstractAPInit):
    def __init__(self, data):
        super().__init__(data)

    def _calculate_origin(self):
        return np.mean(self._data, axis=0)

    def _furthest_point_relative_index(self, current_idata):
        dist_point_to_origin = np.apply_along_axis(
            func1d=lambda point: se_dist(point, self._origin),
            axis=1,
            arr=current_idata[:, 1:])
        return dist_point_to_origin.argmax()

    def _new_cluster(self, label, data):
        return AWardCluster(label, data)
