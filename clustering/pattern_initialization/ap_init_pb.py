from clustering.agglomerative.utils.a_ward_pb_cluster_structure import AWardPBClusterStructure
from clustering.common import minkowski_center
from clustering.pattern_initialization.ap_init import APInit


class APInitPB(APInit):
    """ Anomalous Pattern Initialization with p and beta parameters
    A reference paper: 'A-Ward p b : Effective hierarchical clustering using the Minkowski metric
    and a fast k -means initialisation)', see page 351,
    algorithm Anomalous pattern initialisation for A-Ward p beta and imwk -means p beta"""

    def __init__(self, data, p, beta):
        self._p = p
        self._beta = beta
        super().__init__(data)

    def _create_cluster_structure(self):
        return AWardPBClusterStructure(self._data, self._p, self._beta)

    def _calculate_origin(self):
        return minkowski_center(self._data, self._p)
