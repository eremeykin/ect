import numpy as np
from numpy import linalg as LA
import sys
import time


class Cluster:
    def __init__(self, label, power):
        self.label = label
        self.points_indices = []
        self.centroid = None
        self.power = power

    @staticmethod
    def merge(c1, c2, new_label):
        # print("merge {} with {}".format(c1.label, c2.label))
        assert c1.label != c2.label
        centroid1 = c1.get_centroid()
        n1 = c1.get_power()
        centroid2 = c2.get_centroid()
        n2 = c2.get_power()
        new_centroid = (centroid1 * n1 + centroid2 * n2) / (n1 + n2)
        new_cluster = Cluster(new_label, n1 + n2)
        new_cluster.set_centroid(new_centroid)
        new_cluster.points_indices = c1.points_indices + c2.points_indices
        return new_cluster

    def get_power(self):
        return self.power

    def get_label(self):
        return self.label

    def get_points_indices(self):
        return self.points_indices

    def get_centroid(self):
        return self.centroid

    def set_centroid(self, centroid):
        self.centroid = centroid

    def add_point(self, point_index):
        self.points_indices.append(point_index)

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
    ''' Implements Nearest-neighbor chain algorithm.
    See https://www.revolvy.com/main/index.php?s=Nearest-neighbor%20chain%20algorithm&item_type=topic
    or wikipedia
    '''

    def __init__(self, clusters, K):
        self.clusters = clusters
        self.remaining_clusters = self.clusters[:]
        self.initial_clusters_number = len(clusters)
        self.stack = []
        self.K = K
        self.T = 0
        self.Z = np.empty(shape=(0, 4), dtype=float)

    def _find_nearest(self, base_cluster):
        nearest = None
        min_dist = np.inf
        for cluster in self.remaining_clusters:
            if cluster == base_cluster:
                continue
            dist = base_cluster.award_distance(cluster)
            if dist < min_dist:
                min_dist = dist
                nearest = cluster
        return nearest, min_dist

    def run(self):
        label = self.initial_clusters_number
        while label < 2 * self.initial_clusters_number - 1:
            if not self.stack:
                random_cluster = self.clusters[-1]
                self.stack.append(random_cluster)
            top = self.stack[-1]
            st = time.time()
            nearest, dist = self._find_nearest(top)
            self.T += time.time() - st
            if nearest is None:
                break
            if nearest in self.stack:
                self.Z = np.vstack((self.Z, np.array([top.get_label(), nearest.get_label(), label, dist])))
                self.stack.remove(top)
                self.stack.remove(nearest)
                new_cluster = Cluster.merge(top, nearest, label)
                label += 1
                self.clusters.append(new_cluster)
                self.remaining_clusters.remove(top)
                self.remaining_clusters.remove(nearest)
                self.remaining_clusters.append(new_cluster)
            else:
                self.stack.append(nearest)


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
                new_cluster = Cluster(c, 1)
                self.clusters.append(new_cluster)
            point = data[i, :]
            self.clusters[curr_label].add_point(i)
        for cluster in self.clusters:
            data = self.data[cluster.get_points_indices()]
            cluster.set_centroid(np.mean(data, axis=0))

    def run(self):
        nng = NNGAlogrithm(self.clusters, self.kstar)
        nng.run()
        self.clusters = nng.clusters
        Z = nng.Z
        Zs = Z[Z[:, -1].argsort()]
        pointer = len(Zs) - self.kstar + 1
        Zsl = Zs[:pointer, 0:2]
        result_clusters = []
        while len(result_clusters) < self.kstar:
            # print("consider {}".format(Zs[pointer, 2]))
            if Zs[pointer, 2] not in Zsl:
                result_clusters.append(nng.clusters[int(Zs[pointer,2])])
            pointer-=1
        self.clusters = result_clusters
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


# sys.argv += ["/media/d_disk/projects/Clustering/utils/data15/data1000x15bs.pts"]
# sys.argv += ["/media/d_disk/projects/Clustering/utils/labels15/data1000x15bs.lbs"]
# sys.argv += [4]

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    points_file = sys.argv[1]
    labels_file = sys.argv[2]
    kstar = int(sys.argv[3])
    data = np.loadtxt(points_file)
    labels = np.loadtxt(labels_file, dtype=int)

    start = time.time()
    result = AWard(data, labels, kstar).run()
    print("\n".join([str(x) for x in result]))
