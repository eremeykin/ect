import numpy as np


# TODO Переписать через универсальный класс Direction
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
            print('LABEL='+str(self.label))
        self.data = data
        # TODO normalization ?
        cluster = self.data[[global_indexes]]
        v = data.shape[1]
        cov = np.diag(np.full(v, fill_value=1 / v))
        mean = np.zeros(v)
        dir_vectors = np.random.multivariate_normal(mean, cov, size=_Cluster.s_directions).T
        dir_vectors = dir_vectors / np.linalg.norm(dir_vectors, axis=0)
        P = np.dot(cluster, dir_vectors)  # P[i][j] element is coordinate of point i on j direction
        sort_indexes = np.argsort(P, axis=0)
        # Global Index Matrix
        self.GIM = np.zeros(shape=P.shape, dtype=int)
        for d in range(_Cluster.s_directions):
            self.GIM[:, d] = global_indexes[sort_indexes[:, d]]
        P = np.sort(P, axis=0)
        Pk = np.empty((P.shape[0], 0))  # Pk[i][j] element is kde of coordinate of point i on j direction

        for d in range(_Cluster.s_directions):
            points = P[:, d]
            kde = np.vectorize(KDE(points).kde)
            Pk = np.column_stack((Pk, kde(points)))

        mins = [[] for i in range(_Cluster.s_directions)]
        for r in range(1, len(Pk) - 1):
            left_row = Pk[r - 1]
            row = Pk[r]
            right_row = Pk[r + 1]
            for d in range(_Cluster.s_directions):
                left_value = left_row[d]
                current_value = row[d]
                right_value = right_row[d]
                if left_value > current_value and right_value > current_value:
                    mins[d].append((r, current_value))
        self.epsilon = len([x for x in mins if len(x) > 0]) / _Cluster.s_directions

        print(mins)
        self.split_direction = None
        self.min_ind = None
        self.split_point = None
        for d in range(_Cluster.s_directions):
            for j in range(len(mins[d])):
                if (self.split_direction is None and self.min_ind is None) \
                        or mins[d][j][1] < mins[self.split_direction][self.min_ind][1]:
                    self.split_direction = d
                    self.min_ind = j
                    self.split_point = mins[d][j][0]

    def split(self):
        print('sd'+str(self.split_direction))
        print('sp' + str(self.split_point))
        indexes = self.GIM[:, self.split_direction]
        global_indexes_left = indexes[:self.split_point]
        global_indexes_right = indexes[self.split_point:]
        res = []
        for ind in (global_indexes_left, global_indexes_right):
            if len(ind) > 0:
                res.append(_Cluster(self.data, ind))
        print('return '+str(len(res)))
        return res


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
        print(len(cluster_objs))
    for cluster_obj in cluster_objs:

        labels[[cluster_obj.global_indexes]] = cluster_obj.label
        # tobj.plot(data, labels, prefix='BiKM_R')
    return labels


if __name__ == "__main__":
    from tests.tools.plot import TestObject

    data = np.loadtxt("../tests/data/ikmeans_test3.dat")
    np.random.shuffle(data)

    to = TestObject()
    rs = BiKM_R(data, 0.33, 5, to)
    to.plot(data, rs, show_num=False)
