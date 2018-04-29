import logging
log = logging.getLogger(__name__)
import numpy as np
from time import time

class AWardPB:
    def __init__(self, cluster_structure, k_star=None):
        self._k_star = k_star
        self._cluster_structure = cluster_structure
        self._initial_cluster_count = cluster_structure.clusters_number
        self._completed = False

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

    @property
    def cluster_structure(self):
        if not self._completed:
            raise Exception("Can't return clusters because the algoritm must be "
                            "executed first. Please, use __call__ to run algorithm.")
        return self._cluster_structure

    def __call__(self):
        start =time()
        log.info("starting A-Ward p beta")
        while self._cluster_structure.clusters_number > self._k_star:
            log.info("current clusters number: {:4d}".format(self._cluster_structure.clusters_number))
            clusters = list(self._cluster_structure.clusters)
            i, j = self._nearest(clusters)
            assert i != j
            c1 = clusters[i]
            c2 = clusters[j]
            self._cluster_structure.merge(c1, c2)
        self._completed = True
        log.info("A-Ward p beta completed in {:5.2f} sec.".format(time() - start))
        return self._cluster_structure.current_labels()
