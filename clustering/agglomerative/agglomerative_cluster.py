import numpy as np
from clustering.cluster import Cluster


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
        point = self.data[point_index]
        if self.power == 0:
            self._w = 0
        else:
            assert self._w is not None
            self._w = self.w + 0 + (((self.power * 1) / (self.power + 1)) *
                                    np.sum((self.centroid - point) ** 2))
        super(AgglomerativeCluster, self).add_point(point_index)

    def award_distance(self, cluster):
        na, nb = self.power, cluster.power
        delta = cluster.centroid - self.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance


class WUndefined(BaseException):
    pass
