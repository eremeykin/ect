from clustering.cluster_structure import ClusterStructure
from clustering.agglomerative.utils.a_ward_pb_cluster_structure import AWardPBClusterStructure
import numpy as np
from clustering.common import minkowski_center


class AWardPBClusterStructureMatlabCompatible(AWardPBClusterStructure):
    class Cluster(AWardPBClusterStructure.Cluster):

        def __init__(self, cluster_structure, points_indices, with_setup=True):
            super().__init__(cluster_structure, points_indices, False)
            if with_setup:
                p, beta = cluster_structure.p, cluster_structure.beta
                cluster_points = cluster_structure.data[self._points_indices]
                # update centroid to the component-wise Minkowski centre of all points
                self._centroid = minkowski_center(cluster_points, p)
                # weights update (as per 7)
                D = np.sum(np.abs(cluster_points - self.centroid) ** p, axis=0).astype(np.float64)
                if beta != 1:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        D += 0.01
                        denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
                    isnan = np.isnan(denominator)
                    if np.any(isnan):
                        self._weights = isnan.astype(int) / np.sum(isnan)
                    else:
                        self._weights = np.float64(1.0) / denominator
                else:
                    sh = (cluster_structure.dim_cols,)
                    if np.allclose(D-D[0], np.zeros(sh)):
                        self._weights = np.ones(sh)/sh[0]
                    else:
                        self._weights = np.zeros(shape=sh)
                        self._weights[np.argmin(D)] = 1
                self._is_stable = False
                assert self._weights.shape == (cluster_structure.dim_cols,)
                assert np.abs(np.sum(self._weights) - 1) < 0.0001

    def release_new_cluster(self, points_indices):
        return AWardPBClusterStructureMatlabCompatible.Cluster(self, points_indices)


class IMWKMeansClusterStructureMatlabCompatible(AWardPBClusterStructure):
    def dist_point_to_point(self, point1, point2, cluster_of_point1=None):
        if cluster_of_point1 is None:
            return np.sum((self._equal_weights ** self.beta) * np.abs(point1 - point2) ** self.p) ** (1 / p)  # TODO ATTENTION
        else:
            p = self.p
            beta = self.beta
            return np.sum((cluster_of_point1.weights ** beta) * (np.abs(point1 - point2) ** p)) ** (1 / p)

    def calculate_weights(self, D, mean_D):
        p, beta = self._p, self._beta
        D += mean_D
        if beta != 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                # D += mean_D
                denominator = ((D ** (1 / (beta - 1))) * np.sum((np.float64(1.0) / D) ** (1 / (beta - 1))))
            isnan = np.isnan(denominator)
            if np.any(isnan):
                weights = isnan.astype(int) / np.sum(isnan)
            else:
                weights = np.float64(1.0) / denominator
        else:
            sh = (self.dim_cols,)
            # if np.allclose(D - D[0], np.zeros(sh)):
            #     weights = np.ones(sh) / sh[0]
            # else:
            weights = np.zeros(shape=sh)
            weights[np.argmin(D)] = 1
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
        new_clusters = list()
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
            new_clusters.append(new_cluster)
        return new_clusters
