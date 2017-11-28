import numpy as np
from scipy.spatial.distance import sqeuclidean as se_dist
from clustering.agglomerative.agglomerative_cluster import AWardCluster


class APInit:
    def __init__(self, data):
        self._data = data
        index = np.arange(len(data), dtype=int)[None].T
        self._idata = np.hstack((index, self._data))  # data with index as first column
        self._clusters = []
        self._label_counter = count()
        self._dim_rows = data.shape[0]
        self._dim_cols = data.shape[1]
        self._origin = np.mean(self._data, axis=0)

    def _furthest_point_index(self, current_idata):

        dist_point_to_origin = np.apply_along_axis(
            func1d=lambda point: se_dist(point, self._origin),
            axis=1,
            arr=current_idata[:, 1:])
        return current_idata[:, 0][dist_point_to_origin.argmax()], dist_point_to_origin.argmax()

    def __call__(self):
        current_idata = self._idata
        while len(current_idata) > 0:
            current_index = current_idata[:, 0].astype(int)  # this is index of the points in initial data terms
            current_data = current_idata[:, 1:]  # this is current, cropped data without index

            tentative_centroid_index, tentative_centroid_local_index = self._furthest_point_index(current_idata)  # step 2
            anomalous_cluster = AWardCluster(next(self._label_counter), self._data)
            anomalous_cluster.set_points_and_update(np.array([tentative_centroid_index]))
            anomaly = np.full(shape=current_idata.shape, fill_value=False, dtype=bool)
            anomaly[tentative_centroid_local_index] = True
            while not anomalous_cluster.is_stable():
                dist_point_to_origin = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.distance(point, self._origin),
                    axis=1, arr=current_data)

                dist_point_to_anomalous_centroid = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.distance(point),
                    axis=1, arr=current_data)

                anomaly = dist_point_to_origin >= dist_point_to_anomalous_centroid
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
