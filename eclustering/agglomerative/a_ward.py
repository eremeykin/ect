import numpy as np
from numpy import linalg as LA


class Cluster:
    def __init__(self, cols, label):
        self.data_points = np.full(fill_value=0.0, shape=(1, cols))
        self.label = label
        self.actual_rows = 0
        self.points_indices = []
        self.centroid_changed = True
        self.centroid = None

    @staticmethod
    def merge(c1, c2):
        # print("merge {} with {}".format(c1.label, c2.label))
        assert c1.label != c2.label
        label = c1.label if c1.label < c2.label else c2.label
        d1 = c1.data_points[0:c1.actual_rows, :]
        d2 = c2.data_points[0:c2.actual_rows, :]
        cumulative_data = np.concatenate((d1, d2), axis=0)
        cluster = Cluster(d1.shape[1], label)
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
        for i in range(0, self.dim_rows):
            curr_label = self.labels[i]
            # fill self.clusters with blank clusters
            for c in range(len(self.clusters), curr_label + 1):
                newcluster = Cluster(self.dim_cols, c)
                self.clusters.append(newcluster)
            point = data[i, :]
            self.clusters[curr_label].add_point(i, point)

    def award_distance(self, c1, c2):
        na = c1.get_power()
        nb = c2.get_power()
        centroid1 = c1.get_centroid()
        centroid2 = c2.get_centroid()
        delta = centroid2 - centroid1
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance

    def calculate_distance_matrix(self):
        cnum = len(self.clusters)
        distance_matrix = np.full(fill_value=0.0, shape=(cnum, cnum))
        for i in range(0, cnum):
            for j in range(i + 1, cnum):
                distance_matrix[i, j] = self.award_distance(self.clusters[i],
                                                            self.clusters[j])
        self.distance_matrix = distance_matrix
        self.distance_matrix_init = True

    def find_nearest(self):
        if not self.distance_matrix_init:
            self.calculate_distance_matrix()
        cnum = len(self.clusters)
        min = self.distance_matrix[0, 1]
        c1, c2 = 0, 1
        for i in range(0, cnum):
            for j in range(i + 1, cnum):
                if self.distance_matrix[i, j] < min:
                    min = self.distance_matrix[i, j]
                    c1, c2 = i, j
        return c1, c2

    def run(self):
        k = len(self.clusters)
        while k > self.kstar:
            # print(self.distance_matrix)
            k -= 1
            c1, c2 = self.find_nearest()
            new_cluster = Cluster.merge(self.clusters[c1], self.clusters[c2])
            self.delete_cluster(c2)
            self.update_cluster(c1, new_cluster)
        # form result
        result = np.full(fill_value=0, shape=self.dim_rows)
        for c in range(0, len(self.clusters)):
            clst = self.clusters[c]
            lst = clst.get_points_indices()
            for index in lst:
                result[index] = c
        return result

    def delete_cluster(self, c):
        del self.clusters[c]
        self.distance_matrix = np.delete(self.distance_matrix, c, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, c, axis=1)

    def update_cluster(self, at_index, new_cluster):
        self.clusters[at_index] = new_cluster
        for i in range(0, len(self.clusters)):
            if i < at_index:
                self.distance_matrix[i, at_index] = self.award_distance(new_cluster,
                                                                        self.clusters[i])
            if i > at_index:
                self.distance_matrix[at_index, i] = self.award_distance(new_cluster,
                                                                        self.clusters[i])

    def check_criterion(self, cluster1, cluster2):
        w = lambda clst: ((clst.get_data() - clst.get_centroid()) ** 2).sum()
        delta = lambda c1, c2: w(Cluster.merge(c1, c2)) - w(c1) - w(c2)

        return delta(cluster1, cluster2) < alpha * w(Cluster.merge(cluster1, cluster2))


def a_ward(data, K_star, labels=None):
    return AWard(data, labels, K_star)


if __name__ == "__main__":
    import sys
    import time

    np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    points_file = sys.argv[1]
    labels_file = sys.argv[2]
    kstar = int(sys.argv[3])
    data = np.loadtxt(points_file)
    labels = np.loadtxt(labels_file, dtype=int)

    # start = time.time()
    result = AWard(data, labels, kstar).run()

    print("\n".join([str(x) for x in result]))




    # end = time.time()
    # for i in range(0, len(result)):
    #     print(str(result[i])+" ", end="")
    # print("\ntime:" + str((end - start)))

    # from tests.tools.plot import TestObject
    # data = TestObject.load_data("ikmeans_test7.dat")
    # K_star = 4
    # labels, centroids, weights = a_ward(data, K_star)
