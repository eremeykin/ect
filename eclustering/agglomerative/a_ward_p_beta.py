import numpy as np

from eclustering.pattern_initialization.anomalous_cluster_p_beta import anomalous_cluster_p_beta
from eclustering.common import get_weights, minkowski_center, weighed_minkowski


class _Cluster:
    _last_label = 0

    def __init__(self, data, global_indexes, label, centroid, weights, p, beta):
        self.data = data
        self.label = label
        self.p = p
        self.beta = beta
        self.cord = len(global_indexes)
        if self.label is None:
            _Cluster._last_label += 1
            self.label = _Cluster._last_label
        self.global_indexes = global_indexes
        self.cluster = data[[global_indexes]]
        self.centroid = centroid
        if self.centroid is None:
            self.centroid = minkowski_center(self.cluster, p)
        self.weights = weights
        if self.weights is None:
            self.weights = get_weights(self.cluster, self.centroid, self.p)

    def __str__(self):
        return "Cluster #" + str(self.label) + "(" + str(self.cord) + ")"

    def __repr__(self):
        return str(self)

    @staticmethod
    def distance(cluster1, cluster2):
        Na = cluster1.cord
        Nb = cluster2.cord
        wa = cluster1.weights
        wb = cluster2.weights
        ct_a = cluster1.centroid
        ct_b = cluster2.centroid
        beta = cluster1.beta
        p = cluster1.p
        # print('_______________')
        # print('Na = ' + str(Na))
        # print('Nb = ' + str(Nb))
        # print('wa = ' + str(wa))
        # print('wb = ' + str(wb))
        # print('ct_a = ' + str(ct_a))
        # print('ct_b = ' + str(ct_b))
        # print('beta = ' + str(beta))
        # print('p = ' + str(p))
        return ((Na * Nb) / (Na + Nb)) * np.dot(((wa + wb) / 2) ** beta, np.abs(ct_a - ct_b) ** p)

    @classmethod
    def merge(cls, cluster_obj1, cluster_obj2):
        gi1 = cluster_obj1.global_indexes
        gi2 = cluster_obj2.global_indexes
        l1 = cluster_obj1.label
        l2 = cluster_obj2.label
        return cls(data, np.hstack((gi1, gi2)), min(l1, l2), None, None, cluster_obj1.p, cluster_obj1.beta)


def select_best_clusters(cluster_objs, distance_matrix):
    m = np.argmin(distance_matrix)
    min_a = m // len(distance_matrix)
    min_b = m % len(distance_matrix)
    return max(min_a, min_b), min(min_a, min_b)


def calculate_distance_matrix(cluster_objs):
    K = len(cluster_objs)
    distance_matrix = np.full((K, K), fill_value=np.inf)
    for i in range(0, len(cluster_objs)):
        for j in range(i + 1, len(cluster_objs)):
            distance_matrix[i][j] = _Cluster.distance(cluster_objs[i], cluster_objs[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix


def update_distance_matrix(cluster_objs, distance_matrix, index1, index2):
    # Must delete max index first (due to shift after deletion)
    distance_matrix = np.delete(distance_matrix, index1, 0)  # delete old row from distance matrix
    distance_matrix = np.delete(distance_matrix, index1, 1)  # delete old column from distance matrix
    distance_matrix = np.delete(distance_matrix, index2, 0)  # delete old row from distance matrix
    distance_matrix = np.delete(distance_matrix, index2, 1)  # delete old column from distance matrix
    distance_matrix = np.column_stack((distance_matrix, np.zeros(distance_matrix.shape[0])))
    distance_matrix = np.row_stack((distance_matrix, np.zeros(distance_matrix.shape[1])))
    # distance_matrix.resize(distance_matrix.shape + np.array([1, 1]))  # add row and column for new cluster
    distance_matrix[-1][-1] = np.inf

    for i in range(0, distance_matrix.shape[0] - 1):
        distance_matrix[i][-1] = _Cluster.distance(cluster_objs[i], cluster_objs[-1])
        distance_matrix[-1][i] = distance_matrix[i][-1]
    return distance_matrix


def NEW_a_ward_p_beta(data, p, beta, K_star, labels, centroids, weights, tobj=None):
    if labels is None:
        labels = np.arange(len(data))
        centroids = data
        weights = np.full(shape=data.shape, fill_value=(1 / data.shape[1]))
    cluster_objs = []
    for k in np.unique(labels):
        cluster_objs.append(_Cluster(data, np.where(labels == k)[0], k, centroids[k], weights[k], p, beta))
    distance_matrix = calculate_distance_matrix(cluster_objs)
    while len(cluster_objs) > K_star:
        # print(distance_matrix)
        best_cluster1_index, best_cluster2_index = select_best_clusters(cluster_objs, distance_matrix)
        if best_cluster1_index is None or best_cluster1_index is None:
            break
        best_cluster1 = cluster_objs.pop(best_cluster1_index)
        best_cluster2 = cluster_objs.pop(best_cluster2_index)
        cluster_objs.extend([_Cluster.merge(best_cluster1, best_cluster2)])
        distance_matrix = update_distance_matrix(cluster_objs, distance_matrix, best_cluster1_index,
                                                 best_cluster2_index)
        for cluster_obj in cluster_objs:
            labels[[cluster_obj.global_indexes]] = cluster_obj.label
        tobj.plot(data, labels)
    for cluster_obj in cluster_objs:
        labels[[cluster_obj.global_indexes]] = cluster_obj.label
    return labels


def merge_a_ward_p_beta(data, labels, centroids, weights, a, b, distance, p):
    labels[labels == a] = b  # merge clusters
    # TODO performance leak
    normalize = np.vectorize(lambda x: np.where(np.unique(labels) == x)[0][0])
    labels = normalize(labels)

    # remove old entries
    distance = np.delete(distance, a, 0)  # delete old row from distance matrix
    distance = np.delete(distance, a, 1)  # delete old column from distance matrix
    centroids = np.delete(centroids, a, 0)
    weights = np.delete(weights, a, 0)

    cb = data[labels == b]

    # update centroids and weights
    centroids[b] = minkowski_center(cb, p)
    D = np.sum(np.abs(cb - centroids[b]) ** p, axis=0)
    weights[b] = weights_function(D, D=D, p=p)

    ct_b, Nb, wb = centroids[b], len(cb), weights[b]
    for i in range(len(distance)):
        if i == b:
            distance[i][b] = np.inf
            continue
        ca = data[labels == i]
        ct_a, Na, wa = centroids[i], len(ca), weights[i]
        distance[i][b] = ((Na * Nb) / (Na + Nb)) * np.dot(((wa + wb) / 2) ** beta, np.abs(ct_a - ct_b) ** p)
        distance[b][i] = distance[i][b]
    return distance, labels, centroids, weights


def a_ward_p_beta(data, p, beta, K_star, labels, centroids, weights, tobj=None):
    """Assumed that labels starts with 0 and takes ALL values to max value"""
    K = labels.max() + 1
    distance = np.full((K, K), np.inf)
    for a in range(K):
        for b in range(a, K):
            if a != b:
                ca = data[labels == a]
                cb = data[labels == b]
                ct_a, Na, wa = centroids[a], len(ca), weights[a]
                ct_b, Nb, wb = centroids[b], len(cb), weights[b]
                distance[a][b] = ((Na * Nb) / (Na + Nb)) * np.dot(((wa + wb) / 2) ** beta, np.abs(ct_a - ct_b) ** p)
                distance[b][a] = distance[a][b]
    if tobj is not None: tobj.plot(data, labels, prefix='a_ward_p_beta')

    while K > K_star:
        m = np.argmin(distance)
        min_a = m // len(distance)
        min_b = m % len(distance)
        distance, labels, centroids, weights = merge_a_ward_p_beta(data, labels, centroids, weights,
                                                                   max(min_a, min_b), min(min_a, min_b), distance, p)
        if tobj is not None: tobj.plot(data, labels, prefix='a_ward_p_beta')
        K -= 1
    return labels, centroids, weights


if __name__ == "__main__":
    from tests.tools.plot import TestObject

    data = TestObject.load_data("ikmeans_test3.dat")
    p, beta = 2, 2
    labels, centroids, weights = anomalous_cluster_p_beta(data, p, beta, TestObject('anomalous_cluster_p_beta'))
    # labels, centroids, weights = None, None, None
    tobj = TestObject('a_ward_p_beta')
    labels = NEW_a_ward_p_beta(data, p, beta, 3, labels, centroids, weights, tobj)
    tobj.plot(data, labels, show_num=True)
