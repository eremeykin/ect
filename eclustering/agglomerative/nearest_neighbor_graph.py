import numpy as np


class Cluster:
    def __init__(self, cols, label):
        self.data_points = np.full(fill_value=0.0, shape=(1, cols))
        self.label = label
        self.actual_rows = 0
        self.points_indices = []
        self.centroid_changed = True
        self.centroid = None

    @staticmethod
    def merge(c1, c2, new_label):
        # print("merge {} with {}".format(c1.label, c2.label))
        assert c1.label != c2.label
        d1 = c1.data_points[0:c1.actual_rows, :]
        d2 = c2.data_points[0:c2.actual_rows, :]
        cumulative_data = np.concatenate((d1, d2), axis=0)
        cluster = Cluster(d1.shape[1], new_label)
        cluster.data_points = cumulative_data
        cluster.actual_rows = cumulative_data.shape[0]
        cluster.points_indices = c1.points_indices + c2.points_indices
        cluster.centroid_changed = True
        return cluster

    def get_power(self):
        return self.actual_rows

    def get_label(self):
        return self.label

    def get_data(self):
        return self.data_points[0:self.actual_rows, :]

    def get_points_indices(self):
        return self.points_indices

    def get_centroid(self):
        assert self.actual_rows > 0
        if self.centroid_changed:
            self.centroid = self.data_points[0:self.actual_rows, :].mean(axis=0)
            self.centroid_changed = False
        return self.centroid

    def add_point(self, point_index, point):
        data_rows = self.data_points.shape[0]
        cols = self.data_points.shape[1]
        if self.actual_rows + 1 > data_rows:
            self.data_points = np.append(self.data_points,
                                         np.full(fill_value=0,
                                                 shape=(data_rows * 2, cols)),
                                         axis=0)
        self.data_points[self.actual_rows, :] = point
        self.actual_rows += 1
        self.points_indices.append(point_index)
        self.centroid_changed = True

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return other.label == self.label


class NNGAlogrithm:
    def __init__(self, clusters, K):
        self.clusters = clusters
        self.nearest_clusters = dict()
        for cluster in self.clusters:
            self.nearest_clusters[cluster] = self._find_nearest(cluster)
        self.stack = []
        self.K = K

    def _find_nearest(self, base_cluster):
        nearest = None
        min_dist = np.inf
        for cluster in self.clusters:
            if cluster == base_cluster:
                continue
            dist = base_cluster.a_ward_distance(cluster)
            if dist < min_dist:
                min_dist = dist
                nearest = cluster
        return nearest

    def run(self):
        label = len(self.clusters)
        while len(self.clusters) > self.K:
            if not self.stack:
                random_cluster = self.clusters[-1]
                self.stack.append(random_cluster)
            top = self.stack[-1]
            nearest = self.nearest_clusters[top]
            if nearest in self.stack:
                self.stack.remove(top)
                self.stack.remove(nearest)
                new_cluster = Cluster.merge(top, nearest, label)
                label += 1
                del self.clusters[top]
                del self.clusters[nearest]
                nearest = self._find_nearest(new_cluster, skip=new_cluster_index)
                self.clusters[new_cluster]=nearest
            else:
                self.stack.append(nearest)

