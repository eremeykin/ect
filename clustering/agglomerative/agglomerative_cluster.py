import numpy as np
from abc import ABC, abstractmethod
from clustering.cluster import Cluster
from clustering.common import minkowski_center
from scipy.spatial.distance import sqeuclidean as se_dist


class AgglomerativeCluster(Cluster):
    def __init__(self, label, data):
        super().__init__(label, data)

    @staticmethod
    @abstractmethod
    def merge(cluster1, cluster2, new_label):
        raise NotImplemented("merge of AgglomerativeCluster is not implemented")


class AWardCluster(AgglomerativeCluster):
    """Cluster for A-Ward agglomerative clustering.

    :param int label: integer unique label of this cluster
    :param numpy.array data: of data on which the cluster is defined"""

    def __init__(self, label, data):
        super().__init__(label, data)
        self._w = None

    def set_points(self, points_indices):
        """Assigns points to this cluster WITHOUT update"""

    def _update(self):
        """Updates cluster centroid and w parameter"""
        cluster_points = self._data[self._points_indices]
        # centroid update to the component-wise mean
        self._centroid = np.mean(cluster_points, axis=0)
        # Special property of cluster calculated as sum of square euclidean distance from each point to centroid
        self._w = np.sum((cluster_points - self.centroid) ** 2)
        # weights update (as per 7)
        assert self._centroid.shape == (self._dim_cols,)

    def distance(self, point1, point2=None):
        if point2 is None:
            point2 = self.centroid
        return se_dist(point1, point2)

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
            raise WUndefined("w is undefined for an empty cluster: {}".format(self))
        return self._w

    def award_distance(self, cluster):
        na, nb = self.power, cluster.power
        delta = cluster.centroid - self.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance


class AWardPBCluster(AgglomerativeCluster):
    """Cluster for A-Ward agglomerative clustering with p and beta parameters
    """

    def __init__(self, label, data, p, beta, weights=None):
        """ Constructor of AWardClusterPBeta class.
        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined
        :param float p: Minkowski power
        :param float beta: power of the weights
        :param numpy.array weights: weights of the features in this cluster, shape=[1,dim_cols],
        if None the weights will be initialized with [1/dim_cols, 1/dim_cols, ..., 1/dim_cols] array"""
        super().__init__(label, data)
        self._p = p
        self._beta = beta
        if weights is None:
            self._weights = np.full(shape=(1, self._dim_cols), fill_value=1 / self._dim_cols)
        else:
            raise BaseException("Are you sure what are you doing?")

    def _update(self):
        """Updates cluster centroid and weights"""
        # centroid update to the component-wise Minkowski centre of all points
        cluster_points = self._data[self._points_indices]
        self._centroid = minkowski_center(cluster_points, self._p)
        # weights update (as per 7)
        D = np.sum(np.abs(cluster_points - self.centroid) ** self._p, axis=0).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = ((D ** (1 / (self._p - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (self._p - 1))))
        isnan = np.isnan(denominator)
        if np.any(isnan):
            self._weights = isnan.astype(int) / np.sum(isnan)
        else:
            self._weights = np.float64(1.0) / denominator
        self._is_stable = False
        assert self._weights.shape == (self._dim_cols,)
        assert np.abs(np.sum(self._weights) - 1) < 0.0001

    @staticmethod
    def distance_formula(point1, point2, weights, p, beta):
        return np.sum((weights ** beta) * np.abs(point1 - point2) ** p)

    def distance(self, point1, point2=None):
        """Distance in terms of this cluster. If point2 is None? then distance to centroid"""
        if point2 is None:
            point2 = self.centroid
        return AWardPBCluster.distance_formula(point1, point2, self._weights, self._p, self._beta)

    def merge(self, cluster1, cluster2, new_label):
        raise NotImplemented("merge of AWardPBetaCluster is not implemented")

    def __str__(self):
        res = "AWard_pb Cluster #{}: ".format(self.label)
        res += "centroid: {}, ".format(self.centroid)
        res += "weights: {}, ".format(self._weights)
        res += "power: {}".format(self.power)
        return res


class WUndefined(BaseException):
    pass


class EqualClustersMergeException(BaseException):
    pass


class DifferentDataMergeException(BaseException):
    pass
