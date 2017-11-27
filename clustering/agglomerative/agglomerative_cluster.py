import numpy as np
from abc import ABC, abstractmethod
from clustering.cluster import Cluster
from clustering.common import minkowski_center


class AgglomerativeCluster(ABC, Cluster):
    def __init__(self, label, data):
        super().__init__(label, data)

    @staticmethod
    @abstractmethod
    def merge(cluster1, cluster2, new_label):
        raise NotImplemented("merge of AgglomerativeCluster is not implemented")


class AWardCluster(AgglomerativeCluster):
    """Cluster for A-Ward agglomerative clustering. The add_point method changes the object of
    this class.

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""

    def __init__(self, label, data):
        super().__init__(label, data)
        self._w = None

    @staticmethod
    def merge(cluster1, cluster2, new_label):
        """Merges two clusters to one.
        :param AWardCluster cluster1: first cluster to merge
        :param AWardCluster cluster2: second cluster to merge
        :param int new_label: a label to assign for new cluster"""

        if cluster1 == cluster2:
            raise EqualClustersMergeException("You want to merge a cluster with itself:\n\
            label1 = {}, label2 = {}".format(cluster1.label, cluster2.label))
        if cluster1.data is not cluster2.data:
            raise DifferentDataMergeException("Two clusters are defined on various data sets")

        new_centroid = (cluster1.centroid * cluster1.power + cluster2.centroid * cluster2.power) / (
            cluster1.power + cluster2.power)
        new_cluster = AWardCluster(new_label, cluster1.data)

        new_cluster._points_indices = np.append(cluster1.points_indices, cluster2.points_indices)
        new_cluster._centroid = new_centroid
        new_cluster._w = cluster1.w + cluster2.w + \
                         (((cluster1.power * cluster2.power) / (cluster1.power + cluster2.power)) *
                          np.sum((cluster1.centroid - cluster2.centroid) ** 2))
        return new_cluster

    @property
    def w(self):
        if self._w is None:
            raise WUndefined("w is undefined for an empty cluster: {}" % self)
        return self._w

    def add_point_and_update(self, point_index):
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
        self._points_indices = np.append(self.points_indices, point_index)

    def award_distance(self, cluster):
        na, nb = self.power, cluster.power
        delta = cluster.centroid - self.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance


# TODO think about something like factory
class AWardPBetaCluster(AgglomerativeCluster):
    """Cluster for A-Ward agglomerative clustering with p and beta parameters
    """

    def __init__(self, label, data, p, beta, weights=None, fixed_centroid=False):
        """ Constructor of AWardClusterPBeta class.
        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined
        :param float p: Minkowski power
        :param float beta: power of the weights
        :param numpy.array weights: weights of the features in this cluster, shape=[1,dim_cols],
        if None the weights will be initialized with [1/dim_cols, 1/dim_cols, ..., 1/dim_cols] array,
        :param bool fixed_centroid: do not update centroid, use it for origin cluster"""
        super().__init__(label, data)
        self._p = p
        self._beta = beta
        self._fixed_centroid = fixed_centroid
        self._is_stable = False
        if weights is None:
            self._weights = np.full(shape=(1, self._dim_cols), fill_value=1 / self._dim_cols)

    def is_stable(self):
        return self._is_stable

    def set_points_and_update(self, points_indices):
        """Add points to the cluster and update it. The centroid will be updated if
        fix_centroid is set to False or centroid is None. Weights will be updated anyway.

        :param numpy.array points_indices: list of indices of points to add. Index is based on cluster data."""
        if len(points_indices) == 0 or set(self._points_indices) == set(points_indices):
            self._is_stable = True
            return
        self._points_indices = points_indices
        cluster_points = self._data[self._points_indices]
        if not self._fixed_centroid or self._centroid is None:
            # centroid update to the component-wise Minkowski centre of all points
            self._centroid = minkowski_center(cluster_points, self._p)
        # weights update (as per 7)
        D = np.sum(np.abs(cluster_points - self.centroid) ** self._p, axis=0).astype(np.float64)
        denominator = (D * np.sum((np.float64(1.0) / D) ** (1 / (self._p - 1))))
        isnan = np.isnan(denominator)
        if np.any(isnan):
            self._weights = isnan.astype(int)/np.sum(isnan)
        else:
            self._weights = np.float64(1.0) / denominator
        self._is_stable = False
        assert self._weights.shape == (self._dim_cols,)
        assert np.abs(np.sum(self._weights) - 1) < 0.0001

    def centroid_to_point_distance(self, point):
        return np.sum((self._weights ** self._beta) * np.abs(point - self.centroid) ** self._p)

    def merge(self, cluster1, cluster2, new_label):
        raise NotImplemented("merge of AWardPBetaCluster is not implemented")

    def __str__(self):
        res = "AWard_pb Cluster #{}\n".format(self.label)
        res += "centroid: {}\n".format(self.centroid)
        res += "weights: {}\n".format(self._weights)
        return res

class WUndefined(BaseException):
    pass


class EqualClustersMergeException(BaseException):
    pass


class DifferentDataMergeException(BaseException):
    pass
