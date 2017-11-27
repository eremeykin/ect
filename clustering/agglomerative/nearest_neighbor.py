import numpy as np
from clustering.agglomerative.agglomerative_cluster import AgglomerativeCluster, AWardCluster


class NearestNeighborChain:
    """ Implements nearest-neighbor chain algorithm.
    See https://www.revolvy.com/main/index.php?s=Nearest-neighbor%20chain%20algorithm&item_type=topic
    or wikipedia

    :param list clusters: list of AgglomerativeCluster, initial state of algorithm
    """

    def __init__(self, clusters):
        self._clusters = clusters
        self._remaining_clusters = self._clusters[:]
        self._initial_clusters_number = len(clusters)
        self.merge_matrix = np.empty(shape=(0, 4), dtype=float)

    def _find_nearest(self, base_cluster):
        nearest = None
        min_dist = np.inf
        for cluster in self._remaining_clusters:
            if cluster == base_cluster:
                continue
            dist = base_cluster.award_distance(cluster)
            if dist < min_dist:
                min_dist = dist
                nearest = cluster
        return nearest, min_dist

    def __call__(self):
        """Run nearest-neighbor chain algorithm.
        :returns resulting clusters list and merge matrix.
        Merge matrix is a matrix that consists of 4 columns and number of merges rows
        1,2 columns - two clusters to merge,
        3 column - name of new cluster,
        4 column - AWard distance between clusters to merge
        :rtype: """
        label = self._initial_clusters_number
        stack = []
        while label < 2 * self._initial_clusters_number - 1:
            if not stack:
                random_cluster = self._clusters[-1]
                stack.append(random_cluster)
            top = stack[-1]
            nearest, dist = self._find_nearest(top)
            if nearest is None:
                break
            if nearest in stack:
                self.merge_matrix = np.vstack((self.merge_matrix, np.array([top.label, nearest.label, label, dist])))
                stack.remove(top)
                stack.remove(nearest)
                new_cluster = AWardCluster.merge(top, nearest, label)
                label += 1
                self._clusters.append(new_cluster)
                self._remaining_clusters.remove(top)
                self._remaining_clusters.remove(nearest)
                self._remaining_clusters.append(new_cluster)
            else:
                stack.append(nearest)
        # sort merge_matrix by distance (last column)
        sort_indices = self.merge_matrix[:, -1].argsort()
        self.merge_matrix = self.merge_matrix[sort_indices]
        return self._clusters, self.merge_matrix
