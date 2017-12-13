import numpy as np


class ClusterStructure:
    class Cluster:
        """Base immutable cluster

        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined"""

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            self._points_indices = points_indices
            self._points_indices.flags.writeable = False
            self._indices_tuple = tuple(points_indices)
            self._hash = hash(self._indices_tuple)
            self._centroid = None
            self._cluster_points = cluster_structure.data[points_indices]

        @property
        def points_indices(self):
            """numpy.array of points that are included in this cluster.
            By conventions each index points to one row of data of this cluster."""
            return self._points_indices

        @property
        def cluster_points(self):
            return self._cluster_points

        @property
        def power(self):
            """ Number of points in this cluster"""
            return len(self._indices_tuple)  # I hope it is faster then self._points_indices

        @property
        def centroid(self):
            """Centroid of this cluster"""
            return self._centroid

        def __hash__(self):
            """Computes hash of cluster"""
            return self._hash

        def __eq__(self, other):
            """Compares clusters by it's label only"""
            if type(other) is type(self):
                return self._indices_tuple == other._indices_tuple
            else:
                return False

        def __str__(self):
            """Returns a string representation of cluster in the form of Cluster {#}"""
            return "Cluster: {}".format(self._points_indices)

        def __repr__(self):
            return str(self)

        class CentroidUndefinedException(BaseException):
            """Exception that occurs when centroid of empty cluster is requested"""
            pass

    def __init__(self, data):
        self._clusters = set()
        self._data = data
        self._data.flags.writeable = False
        self._dim_rows = self._data.shape[0]
        self._dim_cols = self._data.shape[1]

    def release_new_batch(self, indices_batch):
        new_clusters = set()
        for indices in indices_batch:
            new_cluster = self.release_new_cluster(indices)
            new_clusters.add(new_cluster)
        return new_clusters

    def release_new_cluster(self, points_indices):
        return ClusterStructure.Cluster(self, points_indices)

    @property
    def dim_cols(self):
        """Cluster structure data features dimension"""
        return self._dim_cols

    @property
    def dim_rows(self):
        """Cluster structure data row dimension"""
        return self._dim_rows

    @property
    def data(self):
        """Data on which cluster structure is defined. It is not writeable."""
        return self._data

    @property
    def clusters_number(self):
        return len(self._clusters)

    @property
    def clusters(self):
        return frozenset(self._clusters)

    def add_cluster(self, cluster):
        self._clusters.add(cluster)

    def add_all_clusters(self, set_of_clusters):
        self._clusters.update(set_of_clusters)

    def del_cluster(self, cluster):
        self._clusters.remove(cluster)

    def clear(self):
        self._clusters = set()

    def current_labels(self):
        """Calculates and returns cluster structure represented by labels of
        each point"""
        labels = np.zeros(self._dim_rows, dtype=int)
        for label, cluster in enumerate(self._clusters):
            labels[cluster.points_indices] = label
        return labels

    class DeleteClusterException(BaseException):
        pass
