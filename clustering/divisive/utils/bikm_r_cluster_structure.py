from clustering.cluster_structure import ClusterStructure
import numpy as np
from clustering.divisive.utils.direction import Direction
from clustering.agglomerative.ik_means.ik_means import IKMeans
from clustering.agglomerative.pattern_initialization.ap_init import APInit


class BiKMeansRClusterStructure(ClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        def __init__(self, cluster_structure, points_indices, with_setup=True):
            super().__init__(cluster_structure, points_indices, with_setup)

        def gen_directions(self, directions_num, mean, cov):
            directions = []
            vectors = np.random.multivariate_normal(mean=mean, cov=cov, size=directions_num)
            for vector in vectors:
                new_direction = Direction(self._cluster_points, vector)
                directions.append(new_direction)
            return directions

    def __init__(self, data, epsilon, directions_num=None, second_chance=False, split_by='ik_means'):
        """split_by = ik_means or max_eps"""
        super().__init__(data)
        self.forbid_split = []
        self._second_chance = second_chance
        self._epsilon = epsilon
        self._split_by = split_by
        self._directions_num = self.dim_cols if directions_num is None else directions_num
        self._mean = np.zeros(shape=self.dim_cols)
        self._cov = np.diag(np.full(fill_value=1 / self.dim_cols, shape=self.dim_cols))

    def _count_directions_with_min(self, directions):
        assert len(directions) == self._directions_num
        return len([d for d in directions if len(d.minima) > 0])

    def find_best_cluster(self):
        best = None
        max_eps_k = 0
        for cluster in self._clusters:
            if cluster in self.forbid_split:
                continue
            directions = cluster.gen_directions(self._directions_num, self._mean, self._cov)
            dirs_with_min = self._count_directions_with_min(directions)
            eps_k = dirs_with_min / self._directions_num
            if eps_k < self._epsilon and not self._second_chance:
                self.forbid_split.append(cluster)
            elif eps_k >= max_eps_k:
                best = cluster
                max_eps_k = eps_k
        return best

    def _run_bisecting_ap_clustering(self, data):
        # normalize data
        mean = np.mean(data, axis=0)
        range_ = np.max(data, axis=0) - np.min(data, axis=0)
        norm_data = (data - mean) / range_
        # run anomalous pattern init
        run_ap_init = APInit(norm_data)
        run_ap_init()
        # get resulting ap clusters, find largest
        cluster_structure = run_ap_init.cluster_structure
        clusters = sorted(list(cluster_structure.clusters), key=lambda x: -x.power)
        largest_ap_clusters = clusters[:2]
        # delete if not in largest
        for c in cluster_structure.clusters:
            if c not in largest_ap_clusters:
                cluster_structure.del_cluster(c)
        # mast have only 2 largest clusters left
        assert len(cluster_structure.clusters) == 2
        # run k-menas
        run_ik_means = IKMeans(cluster_structure)
        run_ik_means()
        final_cluster_structure = run_ik_means.cluster_structure
        assert final_cluster_structure.clusters_number == 2
        # iterate two resulting clusters
        c_iter = iter(final_cluster_structure.clusters)
        first = next(c_iter).points_indices
        second = next(c_iter).points_indices
        return first, second

    def split(self, cluster):
        if self._split_by == "ik_means":
            self.del_cluster(cluster)
            first, second = self._run_bisecting_ap_clustering(cluster.cluster_points)
            index = np.arange(cluster.power, dtype=int)
            left = cluster.points_indices[first]
            right = cluster.points_indices[second]
            left_cluster = BiKMeansRClusterStructure.release_new_cluster(self, left)
            right_cluster = BiKMeansRClusterStructure.release_new_cluster(self, right)
            assert set(left_cluster.points_indices).union(right_cluster.points_indices) == set(cluster.points_indices)
            self.add_cluster(left_cluster)
            self.add_cluster(right_cluster)
            return left_cluster, right_cluster
        if self._split_by == "max_eps":
            raise NotImplemented
        raise AssertionError("unknown split method: {}".format(self._split_by))

    def release_new_cluster(self, points_indices):
        return BiKMeansRClusterStructure.Cluster(self, points_indices)
