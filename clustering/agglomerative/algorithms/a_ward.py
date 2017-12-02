import numpy as np
import sys
from clustering.agglomerative.nearest_neighbor import NearestNeighborChain
from clustering.agglomerative.agglomerative_cluster import AWardCluster


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
        # self._dim_rows = sum((cluster.power for cluster in clusters))
        # self._init_clusters = clusters
        # self._clusters = []
        # self._cluster_by_label = dict()

    def check_criterion(self, cluster1, cluster2, cluster):
        return (1 - self.alpha) * cluster.w < cluster1.w + cluster2.w

    def _get_pointer(self):
        """Get the pointer which specifies where to stop according criteria"""
        if self.k_star is not None:
            pointer = len(self.merge_matrix) - self.k_star + 1
            return pointer
        else:
            for pointer in range(0, len(self.merge_matrix) - 1):
                c_label1, c_label2, c_label_new, dist = self.merge_matrix[pointer]
                c_label1, c_label2, c_label_new = int(c_label1), int(c_label2), int(c_label_new)
                if self.check_criterion(self._cluster_by_label[c_label1],
                                        self._cluster_by_label[c_label2],
                                        self._cluster_by_label[c_label_new]):
                    return pointer

    def __call__(self):
        """Run AWard clustering.

        :returns list of resulting clusters."""

        run_nnc = NearestNeighborChain(self._init_clusters)
        self._clusters, self.merge_matrix = run_nnc()
        self._cluster_by_label = {cluster.label: cluster for cluster in self._clusters}
        # find position where we had to stop according selected criterion: k_star or by formula 8
        pointer = self._get_pointer()
        merged = self.merge_matrix[:pointer, 0:2]  # - all clusters that had been merged before pointer
        # pick to result clusters that are not merged
        result_clusters = []
        for i in range(0, pointer):
            current_cluster = self.merge_matrix[pointer, 2]
            if current_cluster not in merged:
                result_clusters.append(self._cluster_by_label[int(current_cluster)])
            pointer -= 1

        result = np.full(fill_value=0, shape=self._dim_rows)
        for c in range(0, len(result_clusters)):
            cluster = result_clusters[c]
            result[cluster.points_indices] = c
        return np.array(result)
