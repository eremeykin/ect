import numpy as np
from abc import ABC, abstractmethod


class Cluster(ABC):
    """Base immutable cluster

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""

    def __init__(self, label, data):
        if label is None:
            raise BaseException("Wrong! Label can't be None")
        self._label = label
        self._data = data
        self._points_indices = np.empty((0, 1), dtype=int)  # np.array([], dtype=int)
        self._centroid = None
        self._dim_cols = data.shape[1]
        self._is_stable = False

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
            raise CentroidUndefined('Centroid is undefined for empty cluster: {}' % self)
        return self._centroid

    @property
    def data(self):
        """Data on which this cluster is defined"""
        return self._data

    @property
    def is_stable(self):
        """A cluster is stable if it must not be updated after last points assignment"""
        return self._is_stable

    @abstractmethod
    def _update(self):
        """Updates this cluster after points assignment"""
        pass

    @abstractmethod
    def distance(self, point1, point2=None):
        pass

    def set_points_and_update(self, points_indices):
        """Add points to the cluster and update it.

        :param numpy.array points_indices: list of indices of points to add. Index is based on cluster data."""
        if len(points_indices) == 0 or set(self._points_indices) == set(points_indices):
            self._is_stable = True
            return
        self._points_indices = points_indices
        self._update()

    @classmethod
    def clusters_from_labels(cls, data, labels):
        index = np.arange(len(data), dtype=int)
        clusters = []
        if labels is None:
            labels = np.arange(0, len(data), 1)
        for cluster_label in np.unique(labels):
            new_cluster = cls(cluster_label, data)
            new_cluster.set_points_and_update(index[labels == cluster_label])
            clusters.append(new_cluster)
        return clusters


    def __hash__(self):
        """Computes hash of cluster based on it's label only"""
        return hash(self.label)

    def __eq__(self, other):
        """Compares clusters by it's label only"""
        if other is None:
            return False
        return other.label == self.label

    def __str__(self):
        """Returns a string representation of cluster in the form of Cluster {#}"""
        return "Cluster {}".format(self.label)

    def __repr__(self):
        return str(self)

class CentroidUndefined(BaseException):
    """Exception that occurs when centroid of empty cluster is requested"""
    pass
