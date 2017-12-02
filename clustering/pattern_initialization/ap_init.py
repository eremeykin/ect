import numpy as np
from clustering.agglomerative.agglomerative_cluster_structure import AWardClusterStructure


class APInit:
    _MAX_LOOPS = 500

    def __init__(self, data):
        self._data = data
        self._index = np.arange(len(data), dtype=int)[None].T
        self._origin = self._calculate_origin()
        self._completed = False
        self._cluster_structure = self._create_cluster_structure()

    def _create_cluster_structure(self):
        return AWardClusterStructure(self._data)

    def _calculate_origin(self):
        return np.mean(self._data, axis=0)

    def _furthest_point_relative_index(self, current_data):
        dist_point_to_origin = np.apply_along_axis(
            func1d=lambda point: self._cluster_structure.dist_point_to_point(point, self._origin),
            axis=1, arr=current_data)
        return dist_point_to_origin.argmax()

    @property
    def cluster_structure(self):
        if not self._completed:
            raise APInit.AccessToUnavailableResult("Can't return clusters because the algoritm must be "
                                                           "executed first. Please, use __call__ to run algorithm.")
        return self._cluster_structure

    def __call__(self):
        current_data = self._data
        current_index = self._index
        while len(current_index) > 0:
            # step 2
            tentative_centroid_relative_index = self._furthest_point_relative_index(current_data)
            tentative_centroid_index = current_index[tentative_centroid_relative_index]

            anomalous_cluster = self._cluster_structure.make_new_cluster()
            anomalous_cluster.set_points_and_update(tentative_centroid_index)

            anomaly = np.full(shape=current_index.shape, fill_value=False, dtype=bool)
            anomaly[tentative_centroid_relative_index] = True

            for loop_control in range(APInit._MAX_LOOPS):
                dist_point_to_origin = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.dist_point_to_point(point, self._origin),
                    axis=1, arr=current_data)

                dist_point_to_anomalous_centroid = np.apply_along_axis(
                    func1d=lambda point: anomalous_cluster.dist_point_to_cluster(point),
                    axis=1, arr=current_data)

                anomaly = dist_point_to_origin >= dist_point_to_anomalous_centroid
                anomalous_points_indices = current_index[anomaly].flatten()
                anomalous_cluster.set_points_and_update(anomalous_points_indices)  # step 3 and 4,5 inside update
                if anomalous_cluster.is_stable:
                    break
            current_data = current_data[~anomaly]
            current_index = current_index[~anomaly]
        self._completed = True
        return self._cluster_structure.current_labels()

    class AccessToUnavailableResult(BaseException):
        pass
