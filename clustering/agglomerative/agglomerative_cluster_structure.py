import numpy as np
from abc import ABC, abstractmethod
from clustering.common import minkowski_center
from scipy.spatial.distance import sqeuclidean as se_dist
from clustering.cluster_structure import ClusterStructure


class AgglomerativeClusterStructure(ClusterStructure):
    @abstractmethod
    def merge(self, cluster1, cluster2):
        raise NotImplemented("merge of AgglomerativeCluster is not implemented")


class AWardClusterStructure(AgglomerativeClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        """Cluster for A-Ward agglomerative clustering.

        :param int label: integer unique label of this cluster
        :param numpy.array data: of data on which the cluster is defined"""

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            self._w = None  # Do I really need this line?
            super().__init__(cluster_structure, points_indices, with_setup)

        def _setup(self):
            """Updates cluster centroid and w parameter"""
            # centroid update to the component-wise mean
            self._centroid = np.mean(self._cluster_points, axis=0)
            # Special property of cluster calculated as sum of square euclidean distance from each point to centroid
            self._w = np.sum((self._cluster_points - self.centroid) ** 2)
            assert self._centroid.shape == (self.cluster_structure.dim_cols,)

        @classmethod
        def from_params(cls, cluster_structure, points_indices, *args, **kwargs):
            new_cluster = cls(cluster_structure, points_indices, with_setup=False)
            new_cluster._points_indices = points_indices
            new_cluster._w = kwargs['w']
            new_cluster._centroid = kwargs['centroid']
            return new_cluster

        @property
        def w(self):
            return self._w

        def dist_point_to_point(self, point1, point2):
            return se_dist(point1, point2)

        def dist_point_to_cluster(self, point):
            return se_dist(point, self.centroid)

        def dist_cluster_to_cluster(self, cluster):
            """Ward distance between two specified clusters"""
            na, nb = self.power, cluster.power
            delta = self.centroid - cluster.centroid
            distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
            return distance

        class WUndefinedException(BaseException):
            pass

    def dist_point_to_point(self, point1, point2):
        return se_dist(point1, point2)

    def release_new_cluster(self, points_indices):
        return AWardClusterStructure.Cluster(self, points_indices)

    def merge(self, cluster1, cluster2):
        """Merges two clusters to one.
        :param AWardCluster cluster1: the first cluster to merge
        :param AWardCluster cluster2: the second cluster to merge"""

        self.del_cluster(cluster1)
        self.del_cluster(cluster2)

        assert not (cluster1 == cluster2)
        assert cluster1.cluster_structure is cluster2.cluster_structure

        new_centroid = (cluster1.centroid * cluster1.power + cluster2.centroid * cluster2.power) / (
            cluster1.power + cluster2.power)
        new_w = cluster1.w + cluster2.w + \
                (((cluster1.power * cluster2.power) / (cluster1.power + cluster2.power)) *
                 np.sum((cluster1.centroid - cluster2.centroid) ** 2))
        new_points_indices = np.append(cluster1.points_indices, cluster2.points_indices)
        new_cluster = AWardClusterStructure.Cluster.from_params(self, new_points_indices,
                                                                w=new_w, centroid=new_centroid)
        self.add_cluster(new_cluster)
        return new_cluster


class AWardPBClusterStructure(AgglomerativeClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        """Cluster for A-Ward agglomerative clustering with p and beta parameters
        """

        def __init__(self, cluster_structure, points_indices):
            """ Constructor of AWardClusterPBeta class.
            :param int label: integer unique label of this cluster
            :param numpy.array data: of data on which the cluster is defined
            :param numpy.array weights: weights of the features in this cluster, shape=[1,dim_cols],
            if None the weights will be initialized with [1/dim_cols, 1/dim_cols, ..., 1/dim_cols] array"""
            self._weights = None
            super().__init__(cluster_structure, points_indices)

        @classmethod
        def from_params(cls, cluster_structure, points_indices, *args, **kwargs):
            new_cluster = cls(cluster_structure, points_indices, with_setup=False)
            new_cluster._points_indices = points_indices
            new_cluster._weights = kwargs['weights']
            new_cluster._centroid = kwargs['centroid']
            return new_cluster


        def dist_point_to_point(self, point1, point2):
            p = self._cluster_structure.p
            beta = self._cluster_structure.beta
            return np.sum((self.weights ** beta) * np.abs(point1 - point2) ** p)

        def dist_point_to_cluster(self, point):
            return self.dist_point_to_point(point, self.centroid)

        def dist_cluster_to_cluster(self, cluster):
            """WardPB distance between this cluster and specified one"""
            p = self._cluster_structure.p
            beta = self._cluster_structure.beta
            na, nb = self.power, cluster.power
            wa, wb = self.weights, cluster.weights
            delta = self.centroid - cluster.centroid
            weight_multiplier = ((wa + wb) / 2) ** beta
            distance = ((na * nb) / (na + nb)) * (sum(weight_multiplier * (delta ** p)))
            return distance

        @property
        def weights(self):
            return self._weights

        def _setup(self):
            cs = self._cluster_structure
            p, beta = cs.p, cs.beta
            # update centroid to the component-wise Minkowski centre of all points
            self._centroid = minkowski_center(self._cluster_points, p)
            # weights update (as per 7)
            D = np.sum(np.abs(self._cluster_points - self.centroid) ** p, axis=0).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                denominator = ((D ** (1 / (p - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (p - 1))))
            isnan = np.isnan(denominator)
            if np.any(isnan):
                self._weights = isnan.astype(int) / np.sum(isnan)
            else:
                self._weights = np.float64(1.0) / denominator
            assert self._weights.shape == (cs.dim_cols,)
            assert np.abs(np.sum(self._weights) - 1) < 0.0001

        def __str__(self):
            res = "AWard_pb Cluster #{}: ".format(self.label)
            res += "centroid: {}, ".format(self.centroid)
            res += "weights: {}, ".format(self._weights)
            res += "power: {}".format(self.power)
            return res

    def __init__(self, data, p, beta):
        """
        :param float p: Minkowski power
        :param float beta: power of the weights"""
        super().__init__(data)
        self._p = p
        self._beta = beta
        self._equal_weights = np.ones(shape=(self.dim_cols,)) / self.dim_cols

    def dist_point_to_point(self, point1, point2):
        return np.sum((self._equal_weights ** self.beta) * np.abs(point1 - point2) ** self.p)

    @property
    def p(self):
        return self._p

    @property
    def beta(self):
        return self._beta

    def release_new_cluster(self, points_indices):
        return AWardPBClusterStructure.Cluster(self, points_indices)

    def merge(self, cluster1, cluster2):
        """Merges two clusters into one."""
        raise NotImplemented
        # cluster1 = self._clusters_dict[cluster1_label]
        # cluster2 = self._clusters_dict[cluster2_label]
        #
        # self.del_cluster(cluster1_label)
        # self.del_cluster(cluster2_label)
        #
        # assert not (cluster1 == cluster2)
        # assert cluster1.cluster_structure is cluster2.cluster_structure
        #
        # new_centroid = (cluster1.centroid * cluster1.power + cluster2.centroid * cluster2.power) / (
        #     cluster1.power + cluster2.power)
        # new_w = cluster1.w + cluster2.w + \
        #         (((cluster1.power * cluster2.power) / (cluster1.power + cluster2.power)) *
        #          np.sum((cluster1.centroid - cluster2.centroid) ** 2))
        # new_cluster = self.make_new_cluster()
        # # TODO think how can we avoid private members access. Is it legal in this case?
        # new_cluster._points_indices = np.append(cluster1.points_indices, cluster2.points_indices)
        # new_cluster._centroid = new_centroid
        # new_cluster._w = new_w
        # return new_cluster


class AWardPBClusterStructureMatlabCompatible(AWardPBClusterStructure):
    class Cluster(AWardPBClusterStructure.Cluster):

        def _setup(self):
            """Updates cluster centroid and weights"""
            cs = self._cluster_structure
            p, beta = cs.p, cs.beta
            cluster_points = cs.data[self._points_indices]
            # update centroid to the component-wise Minkowski centre of all points
            self._centroid = minkowski_center(cluster_points, p)
            # weights update (as per 7)
            D = np.sum(np.abs(cluster_points - self.centroid) ** p, axis=0).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                D += 0.01
                denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
            isnan = np.isnan(denominator)
            if np.any(isnan):
                self._weights = isnan.astype(int) / np.sum(isnan)
            else:
                self._weights = np.float64(1.0) / denominator
            self._is_stable = False
            assert self._weights.shape == (cs.dim_cols,)
            assert np.abs(np.sum(self._weights) - 1) < 0.0001

    def release_new_cluster(self, points_indices):
        return AWardPBClusterStructureMatlabCompatible.Cluster(self, points_indices)


class IMWKMeansClusterStructureMatlabCompatible(AWardPBClusterStructure):
    class Cluster(AWardPBClusterStructure.Cluster):

        def dist_point_to_point(self, point1, point2):
            p = self._cluster_structure.p
            beta = self._cluster_structure.beta
            return np.sum((self.weights ** beta) * np.abs(point1 - point2) ** p) ** (1 / p)

        def _setup(self):
            """Updates cluster centroid and weights"""
            cs = self._cluster_structure
            p, beta = cs.p, cs.beta
            cluster_points = cs.data[self._points_indices]
            # update centroid to the component-wise Minkowski centre of all points
            self._centroid = minkowski_center(cluster_points, p)
            # weights update (as per 7)
            D = np.sum(np.abs(cluster_points - self.centroid) ** p, axis=0).astype(np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                D += cs.get_mean_D()
                denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
            isnan = np.isnan(denominator)
            if np.any(isnan):
                self._weights = isnan.astype(int) / np.sum(isnan)
            else:
                self._weights = np.float64(1.0) / denominator
            self._is_stable = False
            assert self._weights.shape == (cs.dim_cols,)
            assert np.abs(np.sum(self._weights) - 1) < 0.0001

    # TODO implement it in more subtle way, with cache
    def get_mean_D(self):
        p = self._p
        D = []
        for cluster in self._clusters:
            centroid = minkowski_center(cluster.cluster_points, p)
            d = np.sum(np.abs(cluster.cluster_points - centroid) ** p, axis=0).astype(np.float64)
            D.append(d)
        return np.mean(np.array(D))

    def release_new_cluster(self, points_indices):
        return IMWKMeansClusterStructureMatlabCompatible.Cluster(self, points_indices)
