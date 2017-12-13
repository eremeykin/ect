from clustering.cluster_structure import ClusterStructure
import numpy as np
from clustering.divisive.utils.direction import KDE


class DEPDDPClusterStructure(ClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        def __init__(self, cluster_structure, points_indices, with_setup=True):
            super().__init__(cluster_structure, points_indices, with_setup)
            # normalize cluster points
            mean = np.mean(self._cluster_points, axis=0)
            range_ = np.max(self._cluster_points, axis=0) - np.min(self._cluster_points, axis=0)
            self._norm_points = (self._cluster_points - mean) / range_
            # find principal components
            U, s, Vt = np.linalg.svd(self._norm_points)
            first_pa = Vt[0, :]  # first principal axis
            first_pa = first_pa / np.linalg.norm(first_pa)  # convert to unit vector
            assert np.abs(np.linalg.norm(first_pa) - 1) < 0.0001
            # project cluster points to first principal axis
            first_pa_projections = self._cluster_points @ first_pa
            # reorder _points_indices so that projections to first principal axis are ordered
            projections_sort_indices = np.argsort(first_pa_projections)
            first_pa_projections = first_pa_projections[projections_sort_indices]
            self._points_indices = self._points_indices[projections_sort_indices]
            self._cluster_points = self._cluster_points[projections_sort_indices]
            kde = KDE(first_pa_projections)
            kde_funct = np.vectorize(kde)
            kde_values = kde_funct(first_pa_projections)
            minima = DEPDDPClusterStructure.Cluster._find_minima(kde_values)
            try:
                dmi, dmv = min(minima, key=lambda x: x[1])
                # here we check the only shift +1, because we assign deepest
                # minimum point to the right cluster anyway
                small_shift = (first_pa_projections[dmi + 1] - first_pa_projections[dmi]) / 100  # why exactly 100?
                if kde(first_pa_projections[dmi] + small_shift) < kde(first_pa_projections[dmi]):
                    dmi += 1
                    dmv = kde(first_pa_projections[dmi] + small_shift)
                self.deepest_min_index = dmi
                self.deepest_min_value = dmv
            except ValueError:
                self.deepest_min_index, self.deepest_min_value = None, None

        @staticmethod
        def _find_minima(sorted_kde):
            minima = []
            for i in range(1, len(sorted_kde) - 1):
                left_value = sorted_kde[i - 1]
                current_value = sorted_kde[i]
                right_value = sorted_kde[i + 1]
                if left_value > current_value and right_value > current_value:
                    minima.append((i, current_value))
            return minima

    def find_best_cluster(self):
        try:
            return min([c for c in self._clusters if c.deepest_min_value is not None],
                       key=lambda c: c.deepest_min_value)
        except ValueError:
            return None

    def split(self, cluster):
        self.del_cluster(cluster)
        left_indices = cluster.points_indices[:cluster.deepest_min_index]
        right_indices = cluster.points_indices[cluster.deepest_min_index:]
        left_cluster = DEPDDPClusterStructure.release_new_cluster(self, left_indices)
        right_cluster = DEPDDPClusterStructure.release_new_cluster(self, right_indices)
        self.add_cluster(left_cluster)
        self.add_cluster(right_cluster)
        return left_cluster, right_cluster

    def release_new_cluster(self, points_indices):
        return DEPDDPClusterStructure.Cluster(self, points_indices)