import numpy as np
import sys
import time


class Cluster:
    """ Base cluster"""
    def __init__(self, label, power):
        self._label = label  # unique label of cluster
        self._points_indices = []  # indices of points that are in cluster
        self._centroid = None  # center of cluster
        self._w = None  # w value of cluster, it is calculated as ((cluster.data - cluster.centroid) ** 2).sum()
        self._power = power  # number of points in cluster

    @staticmethod
    def merge(c1, c2, new_label):
        assert c1.label != c2.label
        new_centroid = (c1.centroid * c1.power + c2.centroid * c2.power) / (c1.power + c2.power)
        new_cluster = Cluster(new_label, c1.power + c2.power)
        new_cluster.centroid = new_centroid
        new_cluster.w = c1.w + c2.w + (((c1.power * c2.power) / (c1.power + c2.power)) *
                                       np.sum((c1.centroid - c2.centroid) ** 2))
        new_cluster._points_indices = c1.points_indices + c2.points_indices

        return new_cluster

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    @property
    def power(self):
        return self._power

    @property
    def label(self):
        return self._label

    @property
    def points_indices(self):
        return self._points_indices

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        self._centroid = value

    def add_point(self, point_index):
        self.points_indices.append(point_index)

    def award_distance(self, cluster):
        na, nb = self.power, cluster.power
        delta = cluster.centroid - self.centroid
        distance = ((na * nb) / (na + nb)) * (sum(delta ** 2))
        return distance

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        return other.label == self.label

    def __str__(self):
        return "Cluster {}".format(self.label)

    def __repr__(self):
        return str(self)


class NNGAlogrithm:
    ''' Implements Nearest-neighbor chain algorithm.
    See https://www.revolvy.com/main/index.php?s=Nearest-neighbor%20chain%20algorithm&item_type=topic
    or wikipedia
    '''

    def __init__(self, clusters):
        self.clusters = clusters
        self.remaining_clusters = self.clusters[:]
        self.initial_clusters_number = len(clusters)
        self.stack = []
        # merge_matrix is matrix that consists of 4 columns and #merges rows
        # 1,2 columns - two clusters to merge,
        # 3 column name of new cluster, 4 column - distance between clusters to merge
        self.merge_matrix = np.empty(shape=(0, 4), dtype=float)

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
            nearest, dist = self._find_nearest(top)
            if nearest is None:
                break
            if nearest in self.stack:
                self.merge_matrix = np.vstack((self.merge_matrix, np.array([top.label, nearest.label, label, dist])))
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
        return self.clusters, self.merge_matrix


class AWard:
    def __init__(self, data, labels, k_star=None):
        self.data = data
        self.labels = labels
        self.k_star = k_star
        self.distance_matrix_init = False
        self.dim_rows = data.shape[0]
        self.dim_cols = data.shape[1]
        self.clusters = []
        self.distance_matrix = np.array([])
        self.stack = []
        if self.labels is None:
            self.labels = np.arange(0, len(data), 1)

        # create cluster structure according given labels
        for i in range(0, self.dim_rows):
            curr_label = self.labels[i]
            # fill self.clusters with blank clusters
            for c in range(len(self.clusters), curr_label + 1):
                new_cluster = Cluster(c, 1)
                self.clusters.append(new_cluster)
            self.clusters[curr_label].add_point(i)
        # set centroids and w
        for cluster in self.clusters:
            data = self.data[cluster.points_indices]
            cluster.centroid = np.mean(data, axis=0)
            cluster.w = ((data - cluster.centroid) ** 2).sum()

    @staticmethod
    def check_criterion(cluster1, cluster2, cluster):
        alpha = 0.18
        print('{:10.4} = {:10.4} U {:10.4}; '.format(cluster.w, cluster1.w, cluster2.w), end='')
        print('{:10.4} < {:10.4} : {}'.format(cluster.w - cluster1.w - cluster2.w, alpha * cluster.w,
                                              cluster.w - cluster1.w - cluster2.w < alpha * cluster.w))
        return (1-alpha) * cluster.w < cluster1.w + cluster2.w
        # return cluster.w - cluster1.w - cluster2.w < alpha * cluster.w

    def run(self):
        nng = NNGAlogrithm(self.clusters)
        self.clusters, merge_matrix = nng.run()
        # sort merge matrix by distance (last column)
        sort_indices = merge_matrix[:, -1].argsort()
        merge_matrix = merge_matrix[sort_indices]
        # find position where we had to stop according selected criterion: k_star of by formula 8
        if self.k_star is not None:
            pointer = len(merge_matrix) - self.k_star + 1
        else:
            for pointer in range(0, len(merge_matrix) - 1):
                print(pointer, end='')
                c1, c2, c_new, dist = merge_matrix[pointer]
                c1, c2, c_new = int(c1), int(c2), int(c_new)
                if AWard.check_criterion(self.clusters[c1],
                                         self.clusters[c2],
                                         self.clusters[c_new]):
                    break
        print(pointer)
        merged = merge_matrix[:pointer, 0:2]  # - all clusters that had been merged before pointer
        # pick to result clusters that are not merged
        result_clusters = []
        for i in range(0, pointer):
            # while len(result_clusters) < self.k_star:
            current_cluster = merge_matrix[pointer, 2]
            if current_cluster not in merged:
                result_clusters.append(self.clusters[int(current_cluster)])
            pointer -= 1
        self.clusters = result_clusters
        result = np.full(fill_value=0, shape=self.dim_rows)
        for c in range(0, len(self.clusters)):
            cluster = self.clusters[c]
            for index in cluster.points_indices:
                result[index] = c
        u = np.unique(result)
        d = dict(zip(np.unique(u), np.arange(0, len(u))))
        result = [d[x] for x in result]
        # print(result)
        return result


def a_ward(data, k_star=None, labels=None):
    return AWard(data, labels, k_star).run()


sys.argv += ["/media/d_disk/projects/Clustering/utils/data15/data1000x15bs.pts"]
sys.argv += ["/media/d_disk/projects/Clustering/utils/labels15/data1000x15bs.lbs"]
sys.argv += [4]

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    data = np.loadtxt(sys.argv[1])
    labels = np.loadtxt(sys.argv[2], dtype=int)
    k_star = int(sys.argv[3])
    # k_star = None
    result = AWard(data, labels, k_star).run()
    print("\n".join([str(x) for x in result]))
