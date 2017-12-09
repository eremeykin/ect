import numpy as np


class NearestNeighborChain:
    """ Implements nearest-neighbor chain algorithm.
    See https://www.revolvy.com/main/index.php?s=Nearest-neighbor%20chain%20algorithm&item_type=topic
    or wikipedia

    :param list clusters: list of AgglomerativeCluster, initial state of algorithm
    """

    def __init__(self, cluster_structure):
        self._cs = cluster_structure
        self._clusters = cluster_structure.clusters
        self._initial_clusters_number = cluster_structure.clusters_number
        self._merge_array = []

    def _find_nearest(self, base_cluster):
        nearest = None
        min_dist = np.inf
        for cluster in self._cs.clusters:
            if cluster == base_cluster:
                continue
            dist = self._cs.dist_cluster_to_cluster(base_cluster, cluster)
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
        stack = []
        # while label < 2 * self._initial_clusters_number - 1:
        while self._cs.clusters_number > 1:
            if not stack:
                arbitrary_cluster = next(iter(self._cs.clusters))
                stack.append(arbitrary_cluster)
            top = stack[-1]
            nearest, dist = self._find_nearest(top)
            if nearest is None:
                break
            if nearest in stack:
                merged = self._cs.merge(top, nearest)
                self._merge_array.append([top, nearest, merged, dist])
                stack.remove(top)
                stack.remove(nearest)
            else:
                stack.append(nearest)
        # sort merge_array by distance
        self._merge_array.sort(key=lambda elem: elem[-1])
        return self._merge_array
