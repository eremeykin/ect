import numpy as np

class Cluster:
    """Base cluster

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""
    def __init__(self, label, data):
        assert isinstance(label, int)
        assert isinstance(data, np.ndarray)
        self._label = label
        self._data = data
        self._points_indices = []
        self._centroid = None
        self._dim_cols = data.shape[1]

    @property
    def label(self):
        """A unique label for cluster. It is Integer number."""
        return self._label

    @property
    def points_indices(self):
        """List of points that are included in this cluster.
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

    def __hash__(self):
        """Computes hash of cluster based on it's label only"""
        return hash(self.label)

    def __eq__(self, other):
        """Compares clusters by it's label only"""
        return other.label == self.label

    def __str__(self):
        """Returns a string representation of cluster in the form of Cluster {#}"""
        return "Cluster {}".format(self.label)

    def __repr__(self):
        return str(self)


class CentroidUndefined(BaseException):
    """Exception that occurs when centroid of empty cluster is requested"""
    pass
