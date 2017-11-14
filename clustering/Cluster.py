class Cluster:
    """Base cluster

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""
    def __init__(self, label, data):
        self._label = label
        self._data = data
        self._points_indices = []
        self._centroid = None

    @property
    def label(self):
        """A unique label for cluster. It is Integer number."""
        return self.label

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
            raise CentroidUndefined('This cluster is empty. Centroid undefined : {}' % self)
        return self._centroid

    def add_point(self, point_index):
        """Adds one point to this cluster.
        :param int point_index: index of the point in cluster data to be added

        """
        self.points_indices.append(point_index)

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
