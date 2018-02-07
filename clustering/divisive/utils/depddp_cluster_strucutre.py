from clustering.cluster_structure import ClusterStructure
import numpy as np
from clustering.divisive.utils.direction import Direction


class DEPDDPClusterStructure(ClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        def __init__(self, cluster_structure, points_indices):
            super().__init__(cluster_structure, points_indices)
            # find first principal component
            # 1. normalize cluster points
            mean = np.mean(self._cluster_points, axis=0)
            range_ = np.max(self._cluster_points, axis=0) - np.min(self._cluster_points, axis=0)
            norm_points = (self._cluster_points - mean) / range_
            # 2. make SVD
            U, s, Vt = np.linalg.svd(norm_points)
            first_pa = Vt[0, :]  # first principal axis
            self.first_pa_direction = Direction(self._cluster_points, first_pa)
            reordering = self.first_pa_direction.reordering
            # it's for splitting, where we must know indices correspondence
            self._points_indices = self._points_indices[reordering]
            self._cluster_points = self._cluster_points[reordering]

    def find_best_cluster(self):
        try:
            return min([c for c in self._clusters if c.first_pa_direction.deepest_min_value is not None],
                       key=lambda c: c.first_pa_direction.deepest_min_value)
        except ValueError:
            return None

    def split(self, cluster):
        self.del_cluster(cluster)
        left_indices = cluster.points_indices[:cluster.first_pa_direction.deepest_min_index]
        right_indices = cluster.points_indices[cluster.first_pa_direction.deepest_min_index:]
        left_cluster = DEPDDPClusterStructure.release_new_cluster(self, left_indices)
        right_cluster = DEPDDPClusterStructure.release_new_cluster(self, right_indices)
        self.add_cluster(left_cluster)
        self.add_cluster(right_cluster)
        return left_cluster, right_cluster

    def release_new_cluster(self, points_indices):
        return DEPDDPClusterStructure.Cluster(self, points_indices)
