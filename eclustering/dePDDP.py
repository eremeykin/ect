import numpy as np
import matplotlib.pyplot as plt


def K(x):  # Gaussian density
    return (2 * np.pi) ** (-0.5) * np.e ** (-0.5 * x ** 2)


class Cluster(object):
    _last_label = 0

    def __init__(self, data, global_indexes):
        self.global_indexes = global_indexes
        self.label = Cluster._last_label
        Cluster._last_label += 1
        self.data = data
        cluster = data[[global_indexes]]
        norm = cluster = cluster - cluster.mean(axis=0)
        # principal components
        U, s, Vt = np.linalg.svd(norm)
        S = np.diag(s)
        S.resize(norm.shape)
        p_components = np.dot(U, S)[:, 0]
        sort_indexes = p_components.argsort()
        self.global_indexes = self.global_indexes[sort_indexes]
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
        cluster_obj_l = Cluster(self.data, global_indexes_left)
        cluster_obj_r = Cluster(self.data, global_indexes_right)
        return cluster_obj_l, cluster_obj_r


def select_best_cluster(cluster_objs):
    min_index = None
    for i in range(len(cluster_objs)):
        if cluster_objs[i].min_value:
            if min_index is None or cluster_objs[min_index].min_value > cluster_objs[i].min_value:
                min_index = i
    return min_index


# def select_cluster(clusters_mins):
#     if not clusters_mins:
#         return None
#     index = 0  # the label of cluster with deepest minimum
#     break_point, value = min(clusters_mins[index], key=lambda x: x[1])
#     for curr_index in range(len(clusters_mins)):
#         curr_break_point, curr_value = min(clusters_mins[curr_index], key=lambda x: x[1])
#         if curr_value < value:
#             index = curr_index
#             break_point = curr_break_point
#             value = curr_value
#     return index, break_point
#
#
# def find_min_min(cluster):
#     cluster = cluster - cluster.mean(axis=0)
#     U, s, Vt = np.linalg.svd(cluster)
#     S = np.diag(s)
#     S.resize(cluster.shape)
#     p_components = np.dot(U, S)[:, 0].sort()
#
#     # kernel density estimator
#     n = len(entity_projections)
#     h = np.std(entity_projections) * (4 / (3 * n)) ** (1 / 5)
#
#     def kde(x):
#         s_i = np.apply_along_axis(lambda p: K((x - p) / h), axis=0, arr=entity_projections)
#         return (n * h) ** (-1) * sum(s_i)
#
#     # у него может быть несколько минимумов
#     # нам нучен минимальный минимум по определению
#
#
#     return min()
#
#     values = []
#     for e in p_components:
#         values.append(kde(e))
#     mins = []
#     for j in range(1, len(p_components)) - 1:
#         left_value = values[j - 1]
#         value = values[j]
#         right_value = values[j + 1]
#         if value < left_value and value < right_value:
#             mins.append((j, value))
#     return mins
#
#
# def split_cluster(data, labels, cluster_l, break_point):
#     cluster = data[labels == cluster_l]
#     return labels


def dePDDP(data):
    labels = np.zeros(shape=len(data), dtype=int)
    cluster_objs = [Cluster(data, np.arange(0, len(data), 1, dtype=int))]
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
    to = TestObject()
    # print(dePDDP(data))
    to.plot(data,dePDDP(data),show_num=False)
    # data = data - data.mean(axis=0)
    # U, s, Vt = np.linalg.svd(data)
    # V = Vt.T
    # S = np.diag(s)
    # S.resize(data.shape)
    # pi = 1
    # point = data[pi]
    # print('data\n' + str(data))
    # print('UsV=' + str(U.dot(S.dot(V.T))))
    # print('data*V=\n' + str(data.dot(V)))
    # plt.scatter(data[:, 0], data[:, 1])
    # circle1 = plt.Circle((0, 0), (point[0] ** 2 + point[1] ** 2) ** (0.5), color='b', fill=False)
    # plt.gca().add_artist(circle1)
    # plt.axis('equal')
    # plt.hold(True)
    #
    # pc = get_principal_direction(data)
    # pc2 = get_principal_direction(data, order=2)
    # true_proj = np.inner(point, pc) * pc
    # print((true_proj[0] ** 2 + true_proj[1] ** 2) ** (0.5))
    # plt.plot([0, pc[0]], [0, pc[1]], lw=3)
    # plt.plot([0, pc2[0]], [0, pc2[1]], lw=3)
    # plt.plot([0, U.dot(S)[pi, 0]], [0, U.dot(S)[pi, 1]])
    # plt.scatter(point[0], point[1], s=150, marker='*', color='r')
    #
    # plt.show()
    #
    # exit()
    # pi = 1
    # point = data[pi]
    # print(data)
    # print(point)
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.axis('equal')
    # pc = get_principal_direction(data)
    # pc2 = get_principal_direction(data, order=2)
    # plt.plot([0, pc[0]], [0, pc[1]], lw=3)
    # plt.plot([0, pc2[0]], [0, pc2[1]], lw=3)
    # plt.hold(True)
    # plt.scatter(point[0], point[1], s=150, marker='*', color='r')
    # point_proj = np.inner(point, pc) * pc
    #
    # print('point_proj=' + str(point_proj))
    # S = np.diag(s)
    # S.resize(data.shape)
    #
    # point_proj2 = np.dot(U, S)
    #
    # print('MV=' + str(data.dot(V)))
    # print('point_proj2=' + str(point_proj2[pi]))
    #
    # plt.plot([0, 1 * point_proj[0]], [0, 1 * point_proj[1]])
    # circle1 = plt.Circle((0, 0), (point[0] ** 2 + point[1] ** 2) ** (0.5), color='b', fill=False)
    # plt.gca().add_artist(circle1)
    # for i in (pi,):  # range(len(data)):
    #     plt.plot([0, point_proj2[i, 0]], [0, point_proj2[i, 1]], ls='dashed')
    # perp = point_proj
    # print(np.inner(point_proj - point, pc))
    # plt.show()
    # dePDDP(data)
    # print(get_principal_direction(data))
