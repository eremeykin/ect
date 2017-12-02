from abc import ABC, abstractmethod
import numpy as np
from itertools import count


class ClusterStructure(ABC):
    class Cluster(ABC):
        """Base immutable cluster

        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined"""

        def __init__(self, label, cluster_structure):
            self._label = label
            self._cluster_structure = cluster_structure
            self._points_indices = np.empty((0, 1), dtype=int)  # np.array([], dtype=int)
            self._centroid = None
            self._is_stable = False

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
        def _update(self):
            """Updates this cluster after points assignment"""
            pass

        @property
        def cluster_structure(self):
            return self._cluster_structure

        @property
        def label(self):
            """A unique label for cluster. It is Integer number."""
            return self._label

        @property
        def points_indices(self):
            """numpy.array of points that are included in this cluster.
            By conventions each index points to one row of data of this cluster."""
            return self._points_indices

        @property
        def power(self):
            """ Number of points in this cluster"""
            return len(self._points_indices)

        @property
        def centroid(self):
            """Centroid of this cluster
             :raises CentroidUndefined: if this cluster is empty."""
            if self._centroid is None:
                raise ClusterStructure.Cluster.CentroidUndefinedException(
                    'Centroid is undefined for empty cluster: {}' % self)
            return self._centroid

        @property
        def is_stable(self):
            """A cluster is stable if it must not be updated after last points assignment"""
            return self._is_stable

        # TODO think about points_data instead of points_indices
        def set_points_and_update(self, points_indices):
            """Add points to the cluster and update it.
            -----------------------------------------------------------------------------
            ATTENTION IT SUPPOSES THAT CLUSTER DO NOT DEPEND ON OTHER CLUSTERS AT ALL!!!
            look for example at imwk_means and it's weights update

            :param numpy.array points_indices: list of indices of points to add. Index is based on cluster data."""
            # ATTENTION IT SUPPOSES THAT CLUSTER DO NOT DEPEND ON OTHER CLUSTERS AT ALL!!!
            # look for example at imwk_means and it's weights update
            if set(self._points_indices) == set(points_indices):
                self._is_stable = True
                return
            self._points_indices = points_indices
            self._update()

        def __hash__(self):
            """Computes hash of cluster based on it's label only"""
            return hash(self.label)

        def __eq__(self, other):
            """Compares clusters by it's label only"""
            if type(other) is type(self):
                if other.cluster_structure is not self.cluster_structure:
                    raise BaseException("You defined two different cluster structures and try to compare clusters"
                                        "within it. Are you sure this behaviour is expected?")
                return other.label == self.label
            else:
                return False

        def __str__(self):
            """Returns a string representation of cluster in the form of Cluster {#}"""
            return "Cluster {}".format(self.label)

        def __repr__(self):
            return str(self)

        class CentroidUndefinedException(BaseException):
            """Exception that occurs when centroid of empty cluster is requested"""
            pass

    def __init__(self, data):
        self._clusters_dict = dict()
        self._data = data
        self._dim_rows = self._data.shape[0]
        self._dim_cols = self._data.shape[1]
        self._cluster_labeler = count()

    @abstractmethod
    def _make_new_cluster(self, label):
        pass

    @abstractmethod
    def dist_point_to_point(self, point1, point2):
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
        """Data on which cluster structure is defined"""
        return self._data  # TODO may be I should return a copy? Is it too costly?

    @property
    def clusters_number(self):
        return len(self._clusters_dict)

    @property
    def clusters(self):
        return self._clusters_dict.values()

    def make_new_cluster(self):
        new_label = next(self._cluster_labeler)
        new_cluster = self._make_new_cluster(new_label)
        self._clusters_dict[new_label] = new_cluster
        return new_cluster

    def del_cluster(self, cluster_label):
        if cluster_label in self._clusters_dict:
            del self._clusters_dict[cluster_label]
        else:
            raise ClusterStructure.DeleteClusterException("Can't delete cluster {} "
                                                          "because there is no such cluster in current ")

    def current_labels(self):
        """Calculates and returns cluster structure represented by labels of
        each point"""
        labels = np.zeros(self._dim_rows)
        for label, cluster in self._clusters_dict.items():
            labels[cluster.points_indices] = label
        return labels

    class DeleteClusterException(BaseException):
        pass
