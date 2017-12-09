from clustering.cluster_structure import ClusterStructure
from scipy.spatial.distance import sqeuclidean as se_dist
import numpy as np


class AWardClusterStructure(ClusterStructure):
    """Cluster structure for A-Ward agglomerative algorithm."""

    class Cluster(ClusterStructure.Cluster):
        """Cluster for A-Ward agglomerative clustering."""

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            """Constructor for cluster. Basically the cluster structure generates the cluster, so
            the constructor should be called only from ClusterStructure's methods.

            :param ClusterStructure cluster_structure: a cluster structure which generates the cluster
            :param numpy.array points_indices: [points_in_cluster x 1] the indices of points that forms the cluster.
            Indices are specified based on initial data set."""
            super().__init__(cluster_structure, points_indices, with_setup)
            # centroid update to the component-wise mean
            self._centroid = np.mean(self._cluster_points, axis=0)
            # Special property of cluster calculated as sum of square euclidean distance from each point to centroid
            self._w = np.sum((self._cluster_points - self.centroid) ** 2)
            assert self._centroid.shape == (cluster_structure.dim_cols,)

        @classmethod
        def from_params(cls, cluster_structure, points_indices, **kwargs):
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

    def dist_point_to_point(self, point1, point2, cluster_of_point1=None):
        return se_dist(point1, point2)

    def dist_point_to_cluster(self, point, cluster):
        """Calculates distance from specified point to cluster centroid.
        The distance is equal to squared euclidean distance between this points.

        :param Cluster cluster: a cluster
        :param np.array point: [features] coordinates of the point
        :returns squared euclidean distance"""
        return se_dist(point, cluster.centroid)

    def dist_cluster_to_cluster(self, cluster1, cluster2):
        """Ward distance between this cluster and the specified one

        :param Cluster cluster1: first cluster
        :param Cluster cluster2: second cluster
        :returns Ward distance
        """
        na, nb = cluster1.power, cluster2.power
        delta = cluster1.centroid - cluster2.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance

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
