import numpy as np
from clustering.agglomerative.utils.imwk_means_cluster_structure import IMWKMeansClusterStructure


class IKMeans:
    _MAX_LOOPS = 500

    def __init__(self, cluster_structure):
        self._cluster_structure = cluster_structure
        self._data = cluster_structure.data
        self._completed = False

    @property
    def cluster_structure(self):
        if not self._completed:
            raise BaseException("Not completed yet")
        return self._cluster_structure

    def __call__(self):
        for loop in range(IKMeans._MAX_LOOPS):
            clusters = self._cluster_structure.clusters
            cluster_points = {cluster: [] for cluster in clusters}

            for index, point in enumerate(self._data):
                nearest_cluster = min(clusters, key=lambda c: self._cluster_structure.dist_point_to_cluster(point, c))
                cluster_points[nearest_cluster].append(index)

            new_clusters = self._cluster_structure.release_new_batch(
                [np.array(x) for x in cluster_points.values() if len(x) > 0])  # TODO think how can I write it better
            # stop condition
            if set(new_clusters) == set(clusters):
                break
            self._cluster_structure.clear()
            self._cluster_structure.add_all_clusters(set(new_clusters))
        self._completed = True
        return self.cluster_structure.current_labels()
