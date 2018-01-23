from clustering.agglomerative.pattern_initialization.ap_init_pb_matlab import APInitPBMatlabCompatible
from clustering.agglomerative.utils.matlab_compatible import IMWKMeansClusterStructureMatlabCompatible
from clustering.agglomerative.ik_means.ik_means import IKMeans
from clustering.agglomerative.a_ward_pb import AWardPB
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari


class ChooseP:
    """The Minkowski central partition as a pointer to the distance exponent and consensus partitioning"""

    class ClusteringCriterion:
        def __call__(self, cluster_structure):
            res = 0
            p = cluster_structure.p
            for cluster in cluster_structure.clusters:
                weights = cluster.weights
                points = cluster.cluster_points
                centroid = cluster.centroid
                res += np.sum(weights ** p @ (np.abs(points - centroid) ** p).T)
            return res

    def __init__(self, data, k_star, p_range, times_to_run, criterion=None):
        # algorithm params
        self._data = data
        self._k_star = k_star
        self._p_range = p_range
        self._times_to_run = times_to_run
        self._times_to_run = 1
        self._criterion = criterion
        if self._criterion is None:
            self._criterion = ChooseP.ClusteringCriterion()

    def _single_run(self, p):
        beta = p
        run_ap_init_pb = APInitPBMatlabCompatible(self._data, p, beta)
        run_ap_init_pb()
        # change cluster structure to matlab compatible
        clusters = run_ap_init_pb.cluster_structure.clusters
        new_cluster_structure = IMWKMeansClusterStructureMatlabCompatible(self._data, p, beta)
        new_cluster_structure.add_all_clusters(clusters)
        run_ik_means = IKMeans(new_cluster_structure)
        run_ik_means()
        cs = run_ik_means.cluster_structure
        run_a_ward_pb = AWardPB(cs, self._k_star)
        result = run_a_ward_pb()
        # return result
        return run_a_ward_pb.cluster_structure

    def __call__(self):
        # computing optimal Minkowski partitions
        optimal_minkowski_partitions = []
        for p in self._p_range:
            results = []
            for time in range(self._times_to_run):
                print(time)
                res = self._single_run(p)
                results.append(res)
            best_result = min(results, key=lambda x: self._criterion(x))  # -SW or -CH or clustering criterion
            optimal_minkowski_partitions.append({'p': p, 'labels': best_result.current_labels(),
                                                 'cluster_structure': best_result})
        # computing Minkowski profile
        m = len(optimal_minkowski_partitions)
        ari_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                labels1 = optimal_minkowski_partitions[i]['labels']
                labels2 = optimal_minkowski_partitions[j]['labels']
                value = ari(labels1, labels2)
                ari_matrix[i][j] = value
                ari_matrix[j][i] = value
        profile = np.sum(ari_matrix, axis=0) / m
        # computing the central Minkowski partition
        index_best = np.argmin(profile)
        return self._p_range[index_best], optimal_minkowski_partitions[index_best]['cluster_structure']
