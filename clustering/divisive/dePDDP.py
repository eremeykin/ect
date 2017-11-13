import numpy as np

from clustering.divisive.direction import Direction


class _Cluster(object):
    _last_label = 0

    def __init__(self, data, global_indexes, label=None):
        self.global_indexes = global_indexes
        self.label = label
        if self.label is None:
            _Cluster._last_label += 1
            self.label = _Cluster._last_label
        self.data = data
        cluster = data[[global_indexes]]
        norm = cluster - cluster.mean(axis=0)
        self.cluster = norm
        U, s, Vt = np.linalg.svd(norm)  # principal components
        self.split_direction = Direction(self, Vt[0, :])
        self.min_value = self.split_direction.deepest_min_value

    def split(self):
        s_indexes = self.split_direction.sorted_global_indexes
        global_indexes_left = s_indexes[:self.split_direction.deepest_min_index]
        global_indexes_right = s_indexes[self.split_direction.deepest_min_index:]
        cluster_obj_l = _Cluster(self.data, global_indexes_left, self.label)
        cluster_obj_r = _Cluster(self.data, global_indexes_right)
        return cluster_obj_l, cluster_obj_r


def select_best_cluster(cluster_objs):
    index_best = None
    for i in range(len(cluster_objs)):
        if cluster_objs[i].min_value:
            if index_best is None or cluster_objs[index_best].min_value > cluster_objs[i].min_value:
                index_best = i
    return index_best


def dePDDP(data):
    labels = np.zeros(shape=len(data), dtype=int)
    cluster_objs = [_Cluster(data, np.arange(0, len(data), 1, dtype=int), label=0)]
    while True:
        best_cluster_index = select_best_cluster(cluster_objs)
        if best_cluster_index is None:
            break
        next_cluster_obj = cluster_objs.pop(best_cluster_index)
        cluster_objs.extend(next_cluster_obj.split())
    for cluster_obj in cluster_objs:
        labels[[cluster_obj.global_indexes]] = cluster_obj.label
    return labels


if __name__ == '__main__':
    from tests.tools.plot import TestObject
    data = np.loadtxt("../../tests/data/ikmeans_test12.dat")
    np.random.shuffle(data)
    to = TestObject()
    to.plot(data, dePDDP(data), show_num=False)
