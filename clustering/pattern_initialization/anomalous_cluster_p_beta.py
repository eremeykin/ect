import numpy as np
from clustering.agglomerative.agglomerative_cluster import AWardPBetaCluster
from itertools import count
from clustering.common import minkowski_center


class APInitPB:
    """ Anomalous Pattern Initialization with p and beta parameters
    A reference paper: 'A-Ward p b : Effective hierarchical clustering using the Minkowski metric
    and a fast k -means initialisation)', see page 351,
    algorithm Anomalous pattern initialisation for A-Ward p beta and imwk -means p beta"""

    def __init__(self, data, p, beta):
        self._data = data.astype(object)
        index = np.arange(len(data), dtype=int)[None].T
        self._idata = np.hstack((index, self._data))  # data with index as first column
        self._p = p
        self._beta = beta
        self._clusters = []
        self._label_counter = count()
        self._dim_rows = data.shape[0]
        self._dim_cols = data.shape[1]
        self._origin = minkowski_center(self._data, self._p)
        # self._origin_cluster = AWardPBetaCluster(next(self._label_counter), self._data, self._p, self._beta)

    def _furthest_point_index(self, current_idata):
        equal_weights = np.ones(shape=(self._dim_cols,))/self._dim_cols

        dist_point_to_origin = np.apply_along_axis(
            func1d=lambda point: AWardPBetaCluster.distance_formula(
                point, self._origin, equal_weights, self._p, self._beta),
            axis=1,
            arr=current_idata[:, 1:])
        return current_idata[:, 0][dist_point_to_origin.argmax()]

    def __call__(self):
        current_idata = self._idata
        while len(current_idata) > 0:
            current_index = current_idata[:, 0].astype(int)  # this is index of the points in initial data terms
            current_data = current_idata[:, 1:]  # this is current, cropped data without index

            tentative_centroid_index = self._furthest_point_index(current_idata)  # step 2
            anomalous_cluster = AWardPBetaCluster(next(self._label_counter), self._data, self._p, self._beta)
            anomalous_cluster.set_points_and_update(np.array([tentative_centroid_index]))
            while not anomalous_cluster.is_stable():
                dist_point_to_origin = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.distance(point, self._origin),
                    axis=1, arr=current_data)

                dist_point_to_tentative_centroid = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.distance(point),
                    axis=1, arr=current_data)

                anomaly = dist_point_to_origin >= dist_point_to_tentative_centroid
                anomalous_points_indices = current_index[anomaly]
                anomalous_cluster.set_points_and_update(anomalous_points_indices)  # step 3 and 4,5 inside update
            self._clusters.append(anomalous_cluster)  # step 6
            current_idata = current_idata[~anomaly]

        result = np.full(fill_value=-1, shape=self._dim_rows)
        for c in range(0, len(self._clusters)):
            cluster = self._clusters[c]
            for index in cluster.points_indices:
                result[index] = c
        return np.array(result)


if __name__ == "__main__":
    data = TestObject.load_data("ikmeans_test8.dat")
    tobj = TestObject('anomalous_cluster_p_beta')
    p, beta = 2, 2
    run_anomalous_pattern_init = APInitPB()
    labels, centroids, weights = anomalous_cluster_p_beta(data, p, beta, tobj=tobj)
    tobj.plot(data, labels, centroids=centroeids, prefix="RESULT")
