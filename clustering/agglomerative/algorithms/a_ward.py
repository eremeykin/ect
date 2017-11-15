import numpy as np
import sys
from clustering.agglomerative.nearest_neighbor import NearestNeighborChain
from clustering.agglomerative.agglomerative_cluster import AgglomerativeCluster


class AWard:
    """ Implements AWard clustering algorithm.
    :param np.ndarray data: data to process, columns are features and rows are objects.
    :param list labels: labels for each point in data, which specifies cluster of the point.
    :param int k_star: number of clusters to obtain (optional).
    :param float alpha: parameter of stop criteria (optional). If not specified, k_star is used to stop."""

    def __init__(self, data, labels, k_star=None, alpha=None):
        self.data = data
        self.labels = labels
        self.k_star = k_star
        self.alpha = alpha
        self._dim_rows = data.shape[0]
        self._dim_cols = data.shape[1]
        self._clusters = []

    def _init_clusters(self):
        if self.labels is None:
            self.labels = np.arange(0, len(data), 1)
        # create cluster structure according given labels
        for i in range(0, self._dim_rows):
            curr_label = self.labels[i]
            # fill self.clusters with empty clusters if new label was reached
            # if curr_label < len(self.clusters) - 1, no empty clusters will be created
            for c in range(len(self._clusters), curr_label + 1):
                new_cluster = AgglomerativeCluster(c, self.data)
                self._clusters.append(new_cluster)
            self._clusters[curr_label].add_point(i)

    def check_criterion(self, cluster1, cluster2, cluster):
        return (1 - self.alpha) * cluster.w < cluster1.w + cluster2.w

    def _get_pointer(self):
        if self.k_star is not None:
            pointer = len(self.merge_matrix) - self.k_star + 1
            return pointer
        else:
            for pointer in range(0, len(self.merge_matrix) - 1):
                c1, c2, c_new, dist = self.merge_matrix[pointer]
                c1, c2, c_new = int(c1), int(c2), int(c_new)
                if self.check_criterion(self._clusters[c1],
                                        self._clusters[c2],
                                        self._clusters[c_new]):
                    return pointer

    def __call__(self):
        """Run AWard clustering.

        :returns list of resulting clusters."""
        self._init_clusters()
        run_nnc = NearestNeighborChain(self._clusters)
        self._clusters, self.merge_matrix = run_nnc()
        # find position where we had to stop according selected criterion: k_star or by formula 8
        pointer = self._get_pointer()
        merged = self.merge_matrix[:pointer, 0:2]  # - all clusters that had been merged before pointer
        # pick to result clusters that are not merged
        result_clusters = []
        for i in range(0, pointer):
            current_cluster = self.merge_matrix[pointer, 2]
            if current_cluster not in merged:
                result_clusters.append(self._clusters[int(current_cluster)])
            pointer -= 1
        result = np.full(fill_value=0, shape=self._dim_rows)
        for c in range(0, len(result_clusters)):
            cluster = result_clusters[c]
            for index in cluster.points_indices:
                result[index] = c
        # TODO check the code below. don't know what is it
        u = np.unique(result)
        d = dict(zip(np.unique(u), np.arange(0, len(u))))
        result = [d[x] for x in result]
        return np.array(result)


def a_ward(data, k_star=None, labels=None):
    run_award = AWard(data, labels, k_star)
    return run_award()

sys.argv += ["/home/eremeykin/d_disk/projects/Clustering/utils/data/data10ws.pts"]
sys.argv += ["/home/eremeykin/d_disk/projects/Clustering/utils/labels/data10ws.lbs"]
sys.argv += [4]

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    data = np.loadtxt(sys.argv[1])
    labels = np.loadtxt(sys.argv[2], dtype=int)
    k_star = int(sys.argv[3])
    # k_star = none
    result = a_ward(data, k_star, labels)
    print("\n".join([str(x) for x in result]))
