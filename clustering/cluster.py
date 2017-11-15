import numpy as np

# class ClusterFactory:
#
#     def __init__(self):
#         self.clusters = []
#         pass
#
#     def create_empty_cluster(self):
#         new_cluster = _Cluster()
#
#         return new_cluster

class _Cluster:
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

    def add_point(self, point_index):
        """Adds one point to this cluster.
        :param int point_index: index of the point in cluster data to be added

        """
        point = self.data[point_index]
        if self.power == 0:
            self._centroid = point
        else:
            assert self._centroid is not None
            # TODO Be careful with arbitrary Minkowski power!
            self._centroid = (self.centroid * self.power + point) / (self.power + 1)
        self.points_indices.append(point_index)

    @property
    def data(self):
        """Data on which this cluster is defined"""
        return self._data

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return other.label == self.label

    def __str__(self):
        """Returns a string representation of cluster in the form of Cluster {#}"""
        return "Cluster {}".format(self.label)

    def __repr__(self):
        return str(self)


class CentroidUndefined(BaseException):
    pass
