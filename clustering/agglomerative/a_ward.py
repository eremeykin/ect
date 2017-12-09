import numpy as np
from clustering.agglomerative.utils.a_ward_cluster_structure import AWardClusterStructure

from clustering.agglomerative.utils.nearest_neighbor import NearestNeighborChain


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

    @classmethod
    def from_labels(cls, data, labels, k_star):
        """Constructs the AWard algorithm preset from given labels and k_star"""
        cluster_structure = AWardClusterStructure.from_labels(data, labels)
        return cls(cluster_structure, k_star)

    # def check_criterion(self, cluster1, cluster2, cluster):
    #     return (1 - self._alpha) * cluster.w < cluster1.w + cluster2.w

    def _stop(self, cluster1, cluster2, merged_cluster, clusters):
        return len(clusters) <= self._k_star

    def __call__(self):
        """Run A-Ward clustering, using the preset constructed with object.
        Changes cluster structure, passed to constructor.

        :returns the resulting cluster structure."""
        clusters = set(self._cluster_structure.clusters)
        run_nnc = NearestNeighborChain(self._cluster_structure)
        merge_array = run_nnc()
        for cluster1, cluster2, merged_cluster, dist in merge_array:
            if not self._stop(cluster1, cluster2, merged_cluster, clusters):
                clusters.discard(cluster1)
                clusters.discard(cluster2)
                clusters.add(merged_cluster)
            else:
                break
        self._cluster_structure.clear()
        self._cluster_structure.add_all_clusters(clusters)
        return self._cluster_structure.current_labels()
