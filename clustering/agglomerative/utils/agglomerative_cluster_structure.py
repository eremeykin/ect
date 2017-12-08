import numpy as np
from abc import abstractmethod
from clustering.common import minkowski_center
from scipy.spatial.distance import sqeuclidean as se_dist
from clustering.cluster_structure import ClusterStructure


class AgglomerativeClusterStructure(ClusterStructure):
    """Abstract class for agglomerative algorithms.
    Declares the only method which merges to clusters."""

    @abstractmethod
    def merge(self, cluster1, cluster2):
        """Merge to clusters into one."""
        raise NotImplemented("merge of AgglomerativeCluster is not implemented")


class AWardClusterStructure(AgglomerativeClusterStructure):
    """Cluster structure for A-Ward agglomerative algorithm."""

    class Cluster(ClusterStructure.Cluster):
        """Cluster for A-Ward agglomerative clustering."""

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            """Constructor for cluster. Basically the cluster structure generates the cluster, so
            the constructor should be called only from ClusterStructure's methods.

            :param ClusterStructure cluster_structure: a cluster structure which generates the cluster
            :param numpy.array points_indices: [points_in_cluster x 1] the indices of points that forms the cluster.
            Indices are specified based on initial data set."""
            self._w = None  # Do I really need this line?
            super().__init__(cluster_structure, points_indices, with_setup)

        def _setup(self):
            """Updates cluster centroid and w parameter. This method is called inside __init__."""
            # centroid update to the component-wise mean
            self._centroid = np.mean(self._cluster_points, axis=0)
            # Special property of cluster calculated as sum of square euclidean distance from each point to centroid
            self._w = np.sum((self._cluster_points - self.centroid) ** 2)
            assert self._centroid.shape == (self.cluster_structure.dim_cols,)

        @classmethod
        def from_params(cls, cluster_structure, points_indices, *args, **kwargs):
            """The method allows to create cluster with given parameters.
            The parameters of A-Ward cluster are: w and centroid.

            :param ClusterStructure cluster_structure: cluster structure that generates the cluster
            :param numpy.array points_indices: [points_in_cluster] the indices of points that forms the cluster
            :param float w: value of w cluster specific characteristic
            :param numpy.array centroid: [features] centroids coordinates. One float value for each feature
            :returns new cluster
            """
            new_cluster = cls(cluster_structure, points_indices, with_setup=False)
            new_cluster._points_indices = points_indices
            new_cluster._w = kwargs['w']
            new_cluster._centroid = kwargs['centroid']
            return new_cluster

        @property
        def w(self):
            """Specific characteristic of cluster, calculated
            as sum of squared euclidean distances from each point to centroid.

            :returns w characteristic of cluster"""
            return self._w  # np.sum((self._cluster_points - self.centroid) ** 2)

        def dist_point_to_point(self, point1, point2):
            """Calculates distance from one point to another.
            The distance is equal to squared euclidean distance between this points.

            :param np.array point1: [features] coordinates of one point
            :param np.array point1: [features] coordinates of another point
            :returns squared euclidean distance"""
            return se_dist(point1, point2)

        def dist_point_to_cluster(self, point):
            """Calculates distance from specified point to cluster centroid.
            The distance is equal to squared euclidean distance between this points.

            :param np.array point1: [features] coordinates of the point
            :returns squared euclidean distance"""
            return se_dist(point, self.centroid)

        def dist_cluster_to_cluster(self, cluster):
            """Ward distance between this cluster and the specified one

            :param Cluster cluster: a cluster to calculate distance to
            :returns Ward distance
            """
            na, nb = self.power, cluster.power
            delta = self.centroid - cluster.centroid
            distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
            return distance

        class WUndefinedException(BaseException):
            pass

    @classmethod
    def from_labels(cls, data, labels):
        """Creates cluster structure from given labels of points.

        :param np.array data: [points x features] data set array
        :param labels: [points] labels array. Each value indicates the cluster of appropriate point
        :returns new cluster structure
        """
        result_cs = cls(data)
        index = np.arange(0, len(labels))
        for unique_label in np.unique(labels):
            new_cluster = result_cs.release_new_cluster(index[labels == unique_label])
            result_cs.add_cluster(new_cluster)
        return result_cs

    def dist_point_to_point(self, point1, point2):
        """Calculates distance from one point to another in point of view of the whole cluster structure.

        :param np.array point1: [features] coordinates of one point
        :param np.array point1: [features] coordinates of another point
        :returns squared euclidean distance between points"""
        return se_dist(point1, point2)

    def release_new_cluster(self, points_indices):
        """Creates a new cluster and returns it without adding to the cluster structure.
        But the cluster knows which structure generate it anyway.

        :param numpy.array points_indices: [points_in_cluster] the indices of points that forms the cluster
        :returns new cluster
        """
        return AWardClusterStructure.Cluster(self, points_indices)

    def merge(self, cluster1, cluster2):
        """Merges two clusters to one.

        :param AWardCluster cluster1: the first cluster to merge
        :param AWardCluster cluster2: the second cluster to merge
        :returns  merged cluster
        """

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
    """Cluster structure for A-Ward agglomerative clustering with p and beta parameters"""

    class Cluster(ClusterStructure.Cluster):
        """Cluster for A-Ward agglomerative clustering with p and beta parameters
        """

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            """Constructor for cluster. Basically the cluster structure generates the cluster, so
            the constructor should be called only from ClusterStructure's methods.

            :param ClusterStructure cluster_structure: a cluster structure which generates the cluster
            :param numpy.array points_indices: [points_in_cluster x 1] the indices of points that forms the cluster.
            Indices are specified based on initial data set."""
            self._weights = None
            super().__init__(cluster_structure, points_indices, with_setup)

        @classmethod
        def from_params(cls, cluster_structure, points_indices, *args, **kwargs):
            """The method allows to create cluster with given parameters.
            The parameters of A-Ward cluster are: w and centroid.

            :param ClusterStructure cluster_structure: cluster structure that generates the cluster
            :param numpy.array points_indices: [points_in_cluster] the indices of points that forms the cluster
            :param numpy.array weights: [features] float values of weights for each feature. This weights affects on distances
            :param numpy.array centroid: [features] centroids coordinates. One float value for each feature
            :returns new cluster
            """
            new_cluster = cls(cluster_structure, points_indices, with_setup=False)
            new_cluster._points_indices = points_indices
            new_cluster._weights = kwargs['weights']
            new_cluster._centroid = kwargs['centroid']
            return new_cluster

        def dist_point_to_point(self, point1, point2):
            """Calculates distance from one point to another.
            The distance is equal to squared euclidean distance between this points.

            :param np.array point1: [features] coordinates of one point
            :param np.array point1: [features] coordinates of another point
            :returns distance according current weights"""

            p = self._cluster_structure.p
            beta = self._cluster_structure.beta
            return np.sum((self.weights ** beta) * np.abs(point1 - point2) ** p)

        def dist_point_to_cluster(self, point):
            """Calculates distance from specified point to cluster centroid.
            The distance is calculated according current weights.

            :param np.array point: [features] coordinates of the point
            :returns distance according current weights"""
            return self.dist_point_to_point(point, self.centroid)

        def dist_cluster_to_cluster(self, cluster):
            """WardPB distance between this cluster and specified one
            :param Cluster cluster: cluster to measure distance to
            :returns distance between clusters
            """
            p = self._cluster_structure.p
            beta = self._cluster_structure.beta
            na, nb = self.power, cluster.power
            wa, wb = self.weights, cluster.weights
            delta = np.abs(self.centroid - cluster.centroid)
            weight_multiplier = ((wa + wb) / 2) ** beta
            distance = ((na * nb) / (na + nb)) * (sum(weight_multiplier * (delta ** p)))
            return distance

        @property
        def weights(self):
            """Weights of this cluster

            :return: current weights of cluster
            """
            return self._weights

        def _setup(self):
            """Updates cluster centroid and w parameter. This method is called inside __init__."""
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

    @classmethod
    def from_labels(cls, data, labels, centroids, weights):
        result_cs = cls(data)
        index = np.arange(0, len(labels))
        for l, unique_label in enumerate(np.unique(labels)):
            centroid = centroids[l]
            weights = weights[l]
            new_cluster = cls.Cluster.from_params(result_cs, points_indices=index[labels == unique_label],
                                                  weights=weights, centroid=centroid)
            result_cs.add_cluster(new_cluster)
        return result_cs

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
        p, beta = self._p, self._beta
        self.del_cluster(cluster1)
        self.del_cluster(cluster2)

        assert not (cluster1 == cluster2)  # doesn't work?
        assert cluster1.cluster_structure is cluster2.cluster_structure

        new_points_indices = np.append(cluster1.points_indices, cluster2.points_indices)
        new_points = self._data[new_points_indices]
        new_centroid = minkowski_center(new_points, self.p)

        D = np.sum(np.abs(new_points - new_centroid) ** p, axis=0).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            D += 0.0001
            denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
        isnan = np.isnan(denominator)
        if np.any(isnan):
            new_weights = isnan.astype(int) / np.sum(isnan)
        else:
            new_weights = np.float64(1.0) / denominator

        new_cluster = AWardPBClusterStructure.Cluster.from_params(self, new_points_indices,
                                                                weights=new_weights,
                                                                centroid=new_centroid)
        self.add_cluster(new_cluster)
        return new_cluster

        # raise NotImplemented


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

    def calculate_weights(self, D, mean_D):
        p, beta = self._p, self._beta
        with np.errstate(divide='ignore', invalid='ignore'):
            D += mean_D
            denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
        isnan = np.isnan(denominator)
        if np.any(isnan):
            weights = isnan.astype(int) / np.sum(isnan)
        else:
            weights = np.float64(1.0) / denominator
        assert weights.shape == (self.dim_cols,)
        assert np.abs(np.sum(weights) - 1) < 0.0001
        return weights

    # TODO implement it in more subtle way, with cache
    def get_mean_D(self):
        p, beta = self._p, self._beta
        D = []
        for cluster in self._clusters:
            centroid = minkowski_center(cluster.cluster_points, p)
            d = np.sum(np.abs(cluster.cluster_points - centroid) ** p, axis=0).astype(np.float64)
            D.append(d)
        return np.mean(np.array(D))

    def release_new_cluster(self, points_indices):
        return IMWKMeansClusterStructureMatlabCompatible.Cluster(self, points_indices)

    def release_new_batch(self, indices_batch):
        p, beta = self._p, self._beta
        new_clusters = set()
        centroids = []
        D = []
        for indices in indices_batch:
            cluster_points = self._data[indices]
            centroid = minkowski_center(cluster_points, self.p)
            centroids.append(centroid)
            d = np.sum(np.abs(cluster_points - centroid) ** p, axis=0).astype(np.float64)
            D.append(d)
        D_mean = np.mean(D)
        for i in range(len(indices_batch)):
            points_indices = indices_batch[i]
            weights = self.calculate_weights(D[i], D_mean)
            new_cluster = IMWKMeansClusterStructureMatlabCompatible. \
                Cluster.from_params(self, points_indices, centroid=centroids[i], weights=weights)
            new_clusters.add(new_cluster)
        return new_clusters
