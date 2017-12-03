from abc import ABC, abstractmethod
import numpy as np
from itertools import count


class ClusterStructure(ABC):
    class Cluster(ABC):
        """Base immutable cluster

        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined"""

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            self._cluster_structure = cluster_structure
            self._points_indices = points_indices
            self._points_indices.flags.writeable = False
            self._indices_tuple = tuple(points_indices)
            self._hash = hash(self._indices_tuple)
            if with_setup:
                self._centroid = None
                self._cluster_points = cluster_structure.data[points_indices]
                self._setup()

        @classmethod
        @abstractmethod
        def from_params(cls, cluster_structure, points_indices, *args, **kwargs):
            pass

        @abstractmethod
        def dist_point_to_point(self, point1, point2):
            pass

        @abstractmethod
        def dist_point_to_cluster(self, point):
            pass

        @abstractmethod
        def dist_cluster_to_cluster(self, cluster):
            pass

        @abstractmethod
        def _setup(self):
            pass

        @property
        def cluster_structure(self):
            return self._cluster_structure

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
                # if other.cluster_structure is not self.cluster_structure:
                # raise BaseException("You defined two different cluster structures and try to compare clusters"
                #                     "within it. Are you sure this behaviour is expected?")
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

    @classmethod
    def from_labels(cls, data, labels):
        result_cs = cls(data)
        index = np.arange(0, len(labels))
        for unique_label in np.unique(labels):
            new_cluster = result_cs.release_new_cluster(index[labels == unique_label])
            result_cs.add_cluster(new_cluster)
        return result_cs

    @abstractmethod
    def dist_point_to_point(self, point1, point2):
        pass

    @abstractmethod
    def release_new_cluster(self, points_indices):
        pass

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
        assert cluster.cluster_structure is self
        assert cluster.cluster_structure.data is self.data
        self._clusters.add(cluster)

    def add_all_clusters(self, set_of_clusters):
        # assert all(c.cluster_structure is self for c in set_of_clusters)
        assert all(c.cluster_structure.data is self.data is self.data for c in set_of_clusters)
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
