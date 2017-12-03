import numpy as np
from clustering.agglomerative.nearest_neighbor import NearestNeighborChain
from clustering.agglomerative.agglomerative_cluster_structure import AWardClusterStructure

class AWard:
    """ Implements AWard clustering algorithm.
    :param np.ndarray data: data to process, columns are features and rows are objects.
    :param list labels: labels for each point in data, which specifies cluster of the point.
    :param int k_star: number of clusters to obtain (optional).
    :param float alpha: parameter of stop criteria (optional). If not specified, k_star is used to stop."""

    class WrongParametersException(BaseException):
        pass

    def __init__(self, cluster_structure, k_star=None, alpha=None):
        self._k_star = k_star
        self._alpha = alpha
        self._cluster_structure = cluster_structure
        self._initial_cluster_count = cluster_structure.clusters_number

    @classmethod
    def from_labels(cls, data, labels, k_star=5):
        cluster_structure = AWardClusterStructure.from_labels(data,labels)
        return cls(cluster_structure, k_star)

    # def check_criterion(self, cluster1, cluster2, cluster):
    #     return (1 - self._alpha) * cluster.w < cluster1.w + cluster2.w

    def _stop(self, cluster1, cluster2, merged_cluster, clusters):
        return len(clusters) <= self._k_star

    def __call__(self):
        """Run AWard clustering.

        :returns list of resulting clusters."""
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
