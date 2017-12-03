import numpy as np


class IMWKMeansPB:
    _MAX_LOOPS = 500

    def __init__(self, cluster_structure):
        self._cluster_structure = cluster_structure
        self._data = cluster_structure.data
        self._completed = False

    # TODO implement
    # @staticmethod
    # def from_labels(data, labels):
    #     clusters = AWardPBCluster.clusters_from_labels(data, labels)
    #     return clusters

    @property
    def cluster_structure(self):
        if not self._completed:
            raise BaseException("Not completed yet")
        return self._cluster_structure

    def __call__(self):
        for loop in range(IMWKMeansPB._MAX_LOOPS):
            clusters = self._cluster_structure.clusters
            cluster_points = {cluster: [] for cluster in clusters}

            for index, point in enumerate(self._data):
                nearest_cluster = min(clusters, key=lambda c: c.dist_point_to_cluster(point))
                cluster_points[nearest_cluster].append(index)

            new_clusters = {self._cluster_structure.release_new_cluster(np.array(indices))
                            for indices in cluster_points.values()}
            if new_clusters == clusters:
                break
            clusters = new_clusters
            self._cluster_structure.clear()
            self._cluster_structure.add_all_clusters(clusters)
        self._completed = True
        return self.cluster_structure.current_labels()
