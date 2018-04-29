import numpy as np
from clustering.agglomerative.utils.a_ward_cluster_structure import AWardClusterStructure
from clustering.agglomerative.utils.nearest_neighbor import NearestNeighborChain
from time import time
import logging
log = logging.getLogger(__name__)

class AWard:
    """ Implements AWard clustering algorithm."""

    class WrongParametersException(BaseException):
        pass

    def __init__(self, cluster_structure, k_star=None, alpha=None):
        """ Constructor for A-Ward algorithm. The A-Ward algorithm starts with some cluster structure
        and merges clusters until the specified condition is fulfilled. For example, the condition
        may be specified as number of clusters to obtain.

        :param ClusterStructure cluster_structure: a cluster structure to start with
        :param int k_star: number of clusters to obtain (optional).
        :param float alpha: parameter of stop criteria (optional). If not specified, k_star is used to stop."""
        self._k_star = k_star
        self._alpha = alpha
        self._cluster_structure = cluster_structure
        self._initial_cluster_count = cluster_structure.clusters_number
        self._completed = False

    @property
    def cluster_structure(self):
        if not self._completed:
            raise BaseException("Not completed yet")
        return self._cluster_structure

    def _stop(self, cluster1, cluster2, merged_cluster, clusters):
        if self._alpha is None:
            return len(clusters) <= self._k_star
        else:
            return merged_cluster.w - cluster1.w - cluster2.w >= self._alpha * merged_cluster.w

    def __call__(self):
        """Run A-Ward clustering, using the preset constructed with object.
        Changes cluster structure, passed to constructor.

        :returns the resulting cluster structure."""
        start = time()
        log.info("starting A-Ward algorithm")
        clusters = set(self._cluster_structure.clusters)
        run_nnc = NearestNeighborChain(self._cluster_structure)
        merge_array = run_nnc()
        for cluster1, cluster2, merged_cluster, dist in merge_array:
            if self._stop(cluster1, cluster2, merged_cluster, clusters):
                break
            clusters.discard(cluster1)
            clusters.discard(cluster2)
            clusters.add(merged_cluster)
        self._cluster_structure.clear()
        self._cluster_structure.add_all_clusters(clusters)
        self._completed = True
        log.info("A-Ward completed in {:5.2f} sec.".format(time()-start))
        return self._cluster_structure.current_labels()
