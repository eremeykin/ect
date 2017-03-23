import numpy as np

from eclustering.divisive.direction import Direction
from eclustering.pattern_initialization.anomalous_cluster import anomalous_cluster as ap_init
from sklearn.cluster import KMeans
from collections import Counter as Counter


class _Cluster:
    _last_label = 0
    epsilon = None
    s_directions = None

    def __init__(self, data, global_indexes, label=None):
        self.global_indexes = global_indexes
        self.label = label
        if self.label is None:
            _Cluster._last_label += 1
            self.label = _Cluster._last_label
        self.data = data
        # TODO normalization ?
        self.cluster = self.data[[global_indexes]]
        v = data.shape[1]
        cov = np.diag(np.full(v, fill_value=1 / v))
        mean = np.zeros(v)
        dir_vectors = np.random.multivariate_normal(mean, cov, size=_Cluster.s_directions).T
        self.directions = []
        for d in range(_Cluster.s_directions):
            vector = dir_vectors[:, d]
            self.directions.append(Direction(self, vector))
        self.epsilon = len([d for d in self.directions if d.mins_count > 0]) / _Cluster.s_directions
        self.split_direction = None
        for d in self.directions:
            if d.deepest_min_value is not None:
                if self.split_direction is None:
                    self.split_direction = d
                else:
                    if d.deepest_min_value < self.split_direction.deepest_min_value:
                        self.split_direction = d

    def split_dir(self):
        s_indexes = self.split_direction.sorted_global_indexes
        global_indexes_left = s_indexes[:self.split_direction.deepest_min_index]
        global_indexes_right = s_indexes[self.split_direction.deepest_min_index:]
        cluster_obj_l = _Cluster(self.data, global_indexes_left, self.label)
        cluster_obj_r = _Cluster(self.data, global_indexes_right)
        return cluster_obj_l, cluster_obj_r

    def split(self):
        data = self.data[self.global_indexes]
        labels, centroids = ap_init(data)
        mc = Counter(labels).most_common(2)
        best_centroids = np.array([centroids[mc[0][0]], centroids[mc[1][0]]])
        kmeans = KMeans(n_clusters=2, n_init=1, init=best_centroids).fit(data)

        s_indexes = self.global_indexes
        global_indexes_left = s_indexes[kmeans.labels_ == 0]
        global_indexes_right = s_indexes[kmeans.labels_ == 1]
        cluster_obj_l = _Cluster(self.data, global_indexes_left, self.label)
        cluster_obj_r = _Cluster(self.data, global_indexes_right)
        return cluster_obj_l, cluster_obj_r


def select_best_cluster(cluster_objs):
    index_best = None
    for i in range(len(cluster_objs)):
        if cluster_objs[i].epsilon >= _Cluster.epsilon:
            if index_best is None or cluster_objs[index_best].epsilon > cluster_objs[i].epsilon:
                index_best = i
    return index_best


def BiKM_R(data, epsilon, s_directions, tobj=None):
    labels = np.zeros(shape=len(data), dtype=int)
    _Cluster.epsilon = epsilon
    _Cluster.s_directions = s_directions
    cluster_objs = [_Cluster(data, np.arange(0, len(data), 1, dtype=int), label=0)]
    while True:
        best_cluster_index = select_best_cluster(cluster_objs)
        if best_cluster_index is None:
            break
        next_cluster_obj = cluster_objs.pop(best_cluster_index)
        cluster_objs.extend(next_cluster_obj.split())
        LAB = np.zeros(shape=len(data), dtype=int)
        for cluster_obj in cluster_objs:
            LAB[[cluster_obj.global_indexes]] = cluster_obj.label
        if tobj is not None: tobj.plot(data, LAB, prefix="BiKM_R", show_num=False)
    for cluster_obj in cluster_objs:
        labels[[cluster_obj.global_indexes]] = cluster_obj.label
    return labels


if __name__ == "__main__":
    from tests.tools.plot import TestObject
    data = TestObject.load_data("ikmeans_test12.dat")
    np.random.shuffle(data)
    to = TestObject()
    rs = BiKM_R(data, 0.32, 8, to)
