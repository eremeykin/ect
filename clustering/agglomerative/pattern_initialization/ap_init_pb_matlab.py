from clustering.agglomerative.pattern_initialization.ap_init_pb import APInitPB
from clustering.agglomerative.utils.matlab_compatible import AWardPBClusterStructureMatlabCompatible


class APInitPBMatlabCompatible(APInitPB):
    """ Anomalous Pattern Initialization with p and beta parameters
    A reference paper: 'A-Ward p b : Effective hierarchical clustering using the Minkowski metric
    and a fast k -means initialisation)', see page 351,
    algorithm Anomalous pattern initialisation for A-Ward p beta and imwk -means p beta"""

    def _create_cluster_structure(self):
        return AWardPBClusterStructureMatlabCompatible(self._data, self._p, self._beta)
