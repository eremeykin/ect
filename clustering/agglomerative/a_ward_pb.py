import numpy as np


class AWardPB:
    def __init__(self, cluster_structure, k_star=None):
        self._k_star = k_star
        self._cluster_structure = cluster_structure
        self._initial_cluster_count = cluster_structure.clusters_number

    def _nearest(self, clusters):
        n_clusters = len(clusters)
        distance_matrix = np.full((n_clusters, n_clusters), fill_value=np.inf)
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                c1 = clusters[i]
                c2 = clusters[j]
                distance_matrix[i, j] = self._cluster_structure.dist_cluster_to_cluster(c1, c2)
        minimum = distance_matrix.argmin()  # in flatten array
        min_row = minimum // n_clusters
        min_col = minimum % n_clusters
        assert distance_matrix[min_row, min_col] == distance_matrix.min()
        return min_row, min_col

    def __call__(self):
        while self._cluster_structure.clusters_number > self._k_star:
            clusters = list(self._cluster_structure.clusters)
            i, j = self._nearest(clusters)
            assert i != j
            c1 = clusters[i]
            c2 = clusters[j]
            self._cluster_structure.merge(c1, c2)
        return self._cluster_structure.current_labels()
