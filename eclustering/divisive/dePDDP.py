import numpy as np


def K(x):  # Gaussian density
    return (2 * np.pi) ** (-0.5) * np.e ** (-0.5 * x ** 2)


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
        # principal components
        U, s, Vt = np.linalg.svd(norm)
        S = np.diag(s)
        S.resize(norm.shape)
        p_components = np.dot(U, S)[:, 0]
        sort_indexes = p_components.argsort()
        self.global_indexes = self.global_indexes[sort_indexes]  # most tricky thing
        self.p_components = p_components[sort_indexes]
        # for KDE; DO NOT CHANGE ORDER!
        self.n = len(self.p_components)
        self.h = np.std(self.p_components) * (4 / (3 * self.n)) ** (1 / 5)
        kde_f = np.vectorize(self.kde)
        self.kde_values = kde_f(self.p_components)
        self.min_value = None
        self.min_index = None
        self.find_min()

    def kde(self, x):
        s_i = np.apply_along_axis(lambda p: K((x - p) / self.h), axis=0, arr=self.p_components)
        return (self.n * self.h) ** (-1) * sum(s_i)

    # TODO check whether it is simple minimum
    def find_min(self):
        mins = []
        for i in range(1, len(self.kde_values) - 1):
            left_value = self.kde_values[i - 1]
            current_value = self.kde_values[i]
            right_value = self.kde_values[i + 1]
            if left_value > current_value and right_value > current_value:
                mins.append((i, current_value))
        if mins:
            self.min_index, self.min_value = min(mins, key=lambda x: x[1])

    def split(self):
        global_indexes_left = self.global_indexes[:self.min_index]
        global_indexes_right = self.global_indexes[self.min_index:]
        cluster_obj_l = _Cluster(self.data, global_indexes_left, self.label)
        cluster_obj_r = _Cluster(self.data, global_indexes_right)
        # TODO check if there are no points in new cluster
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

    data = np.loadtxt("../tests/data/ikmeans_test3.dat")
    np.random.shuffle(data)

    to = TestObject()
    to.plot(data, dePDDP(data), show_num=False)
