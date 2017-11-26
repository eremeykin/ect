import numpy as np
from clustering.cluster import Cluster
from clustering.common import minkowski_center


# TODO rename to AWard cluster
class AgglomerativeCluster(Cluster):
    """Cluster for agglomerative clustering

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""

    def __init__(self, label, data):
        super().__init__(label, data)
        self._w = None

    @staticmethod
    def merge(c1, c2, new_label):
        """Merges two clusters to one.
        :param AgglomerativeCluster c1: first cluster to merge
        :param AgglomerativeCluster c2: second cluster to merge
        :param int new_label: a label to assign for new cluster"""
        assert c1.label != c2.label
        assert c1.data is c2.data, " The clusters are not defined on same data. Unable to merge. "
        new_centroid = (c1.centroid * c1.power + c2.centroid * c2.power) / (c1.power + c2.power)
        new_cluster = AgglomerativeCluster(new_label, c1.data)
        new_cluster._points_indices = c1.points_indices + c2.points_indices
        new_cluster._centroid = new_centroid
        new_cluster._w = c1.w + c2.w + (((c1.power * c2.power) / (c1.power + c2.power)) *
                                        np.sum((c1.centroid - c2.centroid) ** 2))
        return new_cluster

    @property
    def w(self):
        if self._w is None:
            raise WUndefined("w is undefined for an empty cluster: {}" % self)
        return self._w

    def add_point(self, point_index):
        """Adds one point to this cluster.
        :param int point_index: index of the point in cluster data to be added

        """
        point = self.data[point_index]
        if self.power == 0:
            self._w = 0
            self._centroid = point
        else:
            assert self._w is not None
            assert self._centroid is not None
            self._w = self.w + 0 + (((self.power * 1) / (self.power + 1)) *
                                    np.sum((self.centroid - point) ** 2))
            self._centroid = (self.centroid * self.power + point) / (self.power + 1)
        self.points_indices.append(point_index)

    def award_distance(self, cluster):
        na, nb = self.power, cluster.power
        delta = cluster.centroid - self.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance


class AgglomerativeClusterPBeta(Cluster):
    """Cluster for Agglomerative clustering with p and beta parameters"""

    def __init__(self, label, data, p, beta, weights=None):
        super().__init__(label, data)
        self.p = p
        self.beta = beta
        if weights is None:
            self._weights = np.full(shape=(1, self._dim_cols), fill_value=1 / self._dim_cols)

    def add_points(self, points_indices):
        self._points_indices.append(points_indices)
        cluster_points = self._data[self._points_indices]
        self._centroid = minkowski_center(cluster_points)
        D = np.sum((cluster_points - self.centroid) ** self.p, axis=0)
        self._weights = 1 / (D * np.sum((1 / D) ** (1 / (self.p - 1))))
        assert np.abs(np.sum(self._weights) - 1) < 0.0001


class WUndefined(BaseException):
    pass
