from clustering.agglomerative.pattern_initialization.ap_init_pb import APInitPB
from clustering.agglomerative.utils.imwk_means_cluster_structure import IMWKMeansClusterStructure
from clustering.agglomerative.ik_means.ik_means import IKMeans
from clustering.agglomerative.a_ward_pb import AWardPB
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from collections import namedtuple
import logging
from time import time
log = logging.getLogger(__name__)


class ChooseP:
    """The Minkowski central partition as a pointer to the distance exponent and consensus partitioning"""

    class AvgSilhouetteWidthCriterion:
        @staticmethod
        def distance(point1, point2):
            return np.sum((point1 - point2) ** 2)

        def _a(self, point_index_i, cluster, cluster_structure):
            data = cluster_structure.data
            dist_list = list()
            for point_index_j in cluster.points_indices:
                # if point_index_i != point_index_j:
                point_i = data[point_index_i]
                point_j = data[point_index_j]
                dist = self.distance(point_i, point_j)
                dist_list.append(dist)
            return np.average(dist_list)

        def _b(self, point_index_i, cluster, cluster_structure):
            data = cluster_structure.data
            avg_list = []
            for curr_cluster in cluster_structure.clusters:
                dist_list = []
                if cluster != curr_cluster:
                    for point_index_j in curr_cluster.points_indices:
                        point_i = data[point_index_i]
                        point_j = data[point_index_j]
                        dist = self.distance(point_i, point_j)
                        dist_list.append(dist)
                    avg_list.append(np.average(dist_list))
            return np.min(avg_list)

        def __call__(self, cluster_structure):
            data = cluster_structure.data
            sw_list = list()
            p = 0
            for cluster in cluster_structure.clusters:
                for point_index in cluster.points_indices:
                    a = self._a(point_index, cluster, cluster_structure)
                    b = self._b(point_index, cluster, cluster_structure)
                    sw = (b - a) / max(b, a)
                    sw_list.append(sw)
                    p+=1
                    log.info("calculating for point: {}/{}".format(p,len(data)))
            return np.average(sw_list)

    def __init__(self, data, k_star, p_range, beta_range):
        self._data = data
        self._k_star = k_star
        self._p_range = p_range
        self._beta_range = beta_range
        self._criterion = ChooseP.AvgSilhouetteWidthCriterion()

    def _single_run(self, p, beta):
        run_ap_init_pb = APInitPB(self._data, p, beta)
        run_ap_init_pb()
        # change cluster structure to matlab compatible
        clusters = run_ap_init_pb.cluster_structure.clusters
        new_cluster_structure = IMWKMeansClusterStructure(self._data, p, beta)
        new_cluster_structure.add_all_clusters(clusters)
        run_ik_means = IKMeans(new_cluster_structure)
        run_ik_means()
        cs = run_ik_means.cluster_structure
        run_a_ward_pb = AWardPB(cs, self._k_star)
        result = run_a_ward_pb()
        return run_a_ward_pb.cluster_structure

    def __call__(self):
        best_cluster_structure, best_p, best_beta = None, None, None
        max_criterion = -np.inf
        criterion_matrix = np.zeros((len(self._p_range), len(self._beta_range)))
        for row, p in enumerate(self._p_range):
            for col, beta in enumerate(self._beta_range):
                print(p, beta)
                from time import time
                st = time()
                res_cluster_structure = self._single_run(p, beta)
                labels = res_cluster_structure.current_labels()
                criterion_value = self._criterion(res_cluster_structure)
                criterion_matrix[row, col] = criterion_value
                if criterion_value > max_criterion:
                    max_criterion = criterion_value
                    best_cluster_structure = res_cluster_structure
                    best_p, best_beta = p, beta
                print(time()-st)
        return best_p, best_beta, best_cluster_structure, criterion_matrix
