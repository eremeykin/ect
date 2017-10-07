import numpy as np
from numpy import linalg as LA
import sys
import time


class Cluster:
    def __init__(self, cols, label, centroid):
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

    def award_distance(self, cluster):
        na = self.get_power()
        nb = cluster.get_power()
        centroid1 = self.get_centroid()
        centroid2 = cluster.get_centroid()
        delta = centroid2 - centroid1
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return other.label == self.label

    def __str__(self):
        return "Cluster {}".format(self.get_label())

    def __repr__(self):
        return str(self)


class NNGAlogrithm:
    def __init__(self, clusters, K):
        self.clusters = clusters
        self.nearest_clusters = dict()
        for cluster in self.clusters:
            self.nearest_clusters[cluster] = self._find_nearest(cluster)
        self.stack = []
        self.K = K
        self.T = 0
        self.Z = np.empty(shape=(0, 4), dtype=float)

    def _find_nearest(self, base_cluster):
        nearest = None
        min_dist = np.inf
        for cluster in self.clusters:
            if cluster == base_cluster:
                continue
            dist = base_cluster.award_distance(cluster)
            if dist < min_dist:
                min_dist = dist
                nearest = cluster
        return nearest, min_dist

    def run(self):
        label = len(self.clusters)
        # print(self.K)
        while len(self.clusters) > 0:
            # print(self.clusters)
            # for cluster in self.clusters:
            #     print("{}: {}".format(cluster, cluster.points_indices))

            if not self.stack:
                random_cluster = self.clusters[-1]
                self.stack.append(random_cluster)
            top = self.stack[-1]
            st = time.time()
            nearest, dist = self._find_nearest(top)  # self.nearest_clusters[top]
            self.T += time.time() - st
            if nearest is None:
                break
            if nearest in self.stack:
                self.Z = np.vstack((self.Z,np.array([top.get_label(), nearest.get_label(), label, dist])))
                self.stack.remove(top)
                self.stack.remove(nearest)
                print("merge {} with {}".format(top.get_label(), nearest.get_label()))
                new_cluster = Cluster.merge(top, nearest, label)
                label += 1
                self.clusters.remove(top)
                self.clusters.remove(nearest)
                del self.nearest_clusters[top]
                del self.nearest_clusters[nearest]
                nearest = self._find_nearest(new_cluster)
                self.nearest_clusters[new_cluster] = nearest
                self.clusters.append(new_cluster)
            else:
                self.stack.append(nearest)
        print(self.T)


class AWard:
    def __init__(self, data, labels, kstar):
        self.data = data
        self.labels = labels
        self.kstar = kstar
        self.distance_matrix_init = False
        self.dim_rows = data.shape[0]
        self.dim_cols = data.shape[1]
        self.clusters = []
        self.distance_matrix = np.array([])
        self.stack = []
        for i in range(0, self.dim_rows):
            curr_label = self.labels[i]
            # fill self.clusters with blank clusters
            for c in range(len(self.clusters), curr_label + 1):
                new_cluster = Cluster(self.dim_cols, c)
                self.clusters.append(new_cluster)
            point = data[i, :]
            self.clusters[curr_label].add_point(i, point)

    def run(self):
        nng = NNGAlogrithm(self.clusters, self.kstar)
        nng.run()
        self.clusters = nng.clusters
        print(nng.Z)
        # print(self.T)
        result = np.full(fill_value=0, shape=self.dim_rows)
        for c in range(0, len(self.clusters)):
            clst = self.clusters[c]
            lst = clst.get_points_indices()
            for index in lst:
                result[index] = c
        return result

    def check_criterion(self, cluster1, cluster2):
        w = lambda clst: ((clst.get_data() - clst.get_centroid()) ** 2).sum()
        delta = lambda c1, c2: w(Cluster.merge(c1, c2)) - w(c1) - w(c2)

        return delta(cluster1, cluster2) < alpha * w(Cluster.merge(cluster1, cluster2))


def a_ward(data, K_star, labels=None):
    return AWard(data, labels, K_star).run()


sys.argv += ["/media/d_disk/projects/Clustering/utils/data15/data10x15bs.pts"]
sys.argv += ["/media/d_disk/projects/Clustering/utils/labels15/data10x15bs.lbs"]
sys.argv += [4]

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    points_file = sys.argv[1]
    labels_file = sys.argv[2]
    kstar = int(sys.argv[3])
    data = np.loadtxt(points_file)
    labels = np.loadtxt(labels_file, dtype=int)

    start = time.time()
    result = AWard(data, labels, kstar).run()
    print(time.time() - start)
    print("\n".join([str(x) for x in result]))



    # end = time.time()
    # for i in range(0, len(result)):
    #     print(str(result[i])+" ", end="")
    # print("\ntime:" + str((end - start)))

    # from tests.tools.plot import TestObject
    # data = TestObject.load_data("ikmeans_test7.dat")
    # K_star = 4
    # labels, centroids, weights = a_ward(data, K_star)
