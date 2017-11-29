from clustering.agglomerative.agglomerative_cluster import AWardPBCluster
import numpy as np


class IMWKMeansPB:
    def __init__(self, clusters, p, beta):
        self._p = p
        self._beta = beta
        self._clusters = clusters
        self._dim_rows = sum((cluster.power for cluster in clusters))
        if self._clusters:
            self._data = self._clusters[0].data.astype(object)  # TODO think about something more elegant
            index = np.arange(len(self._data), dtype=int)[None].T
            self._idata = np.hstack((index, self._data))  # data with index as first column

    @staticmethod
    def from_labels(data, labels):
        clusters = AWardPBCluster.clusters_from_labels(data, labels)
        return clusters

    def _new_cluster(self, label, data):
        return AWardPBCluster(label, data, self._p, self._beta)

    def __call__(self):
        # entity assignment
        while True:
            cluster_points = {cluster.label: [] for cluster in self._clusters}

            def distribute_points(ipoint):
                index, point = ipoint[0], ipoint[1:]
                cluster = min(self._clusters, key=lambda clstr: clstr.distance(point))
                cluster_points[cluster.label].append(index)

            np.apply_along_axis(func1d=distribute_points, axis=1, arr=self._idata)
            for cluster in self._clusters:
                label = cluster.label
                cluster.set_points_and_update(cluster_points[label])
            if all((cluster.is_stable for cluster in self._clusters)):
                break

        result = np.full(fill_value=0, shape=self._dim_rows)
        for c in range(0, len(self._clusters)):
            cluster = self._clusters[c]
            result[cluster.points_indices] = c
        return np.array(result)
