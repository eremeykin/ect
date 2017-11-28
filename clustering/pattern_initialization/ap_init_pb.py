import numpy as np
from clustering.agglomerative.agglomerative_cluster import AWardPBetaCluster
from clustering.pattern_initialization.abstract_ap_init import AbstractAPInit
from itertools import count
from clustering.common import minkowski_center


class APInitPB(AbstractAPInit):
    """ Anomalous Pattern Initialization with p and beta parameters
    A reference paper: 'A-Ward p b : Effective hierarchical clustering using the Minkowski metric
    and a fast k -means initialisation)', see page 351,
    algorithm Anomalous pattern initialisation for A-Ward p beta and imwk -means p beta"""

    class APInitPBClusterFactory(AbstractAPInit.AbstractClusterFactory):
        def __init__(self, p, beta):
            self._p = p
            self._beta = beta

        def new(self, label, data):
            return AWardPBetaCluster(label, data, self._p, self._beta)

    def __init__(self, data, p, beta):
        self._p = p
        self._beta = beta
        super().__init__(data)

    def _calculate_origin(self):
        return minkowski_center(self._data, self._p)

    def _furthest_point_relative_index(self, current_idata):
        equal_weights = np.ones(shape=(self._dim_cols,)) / self._dim_cols

        dist_point_to_origin = np.apply_along_axis(
            func1d=lambda point: AWardPBetaCluster.distance_formula(
                point, self._origin, equal_weights, self._p, self._beta),
            axis=1,
            arr=current_idata[:, 1:])
        return dist_point_to_origin.argmax()

    def _get_cluster_factory(self):
        return APInitPB.APInitPBClusterFactory(self._p, self._beta)
