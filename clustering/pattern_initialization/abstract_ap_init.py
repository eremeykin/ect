import numpy as np
from itertools import count
from abc import ABC, abstractmethod


class AbstractAPInit(ABC):

    class AbstractClusterFactory(ABC):

        @abstractmethod
        def new(self, label, data):
            """Returns a new cluster with given label and based of specified data."""
            pass

    def __init__(self, data):
        self._data = data.astype(object)  # TODO think about something more elegant
        index = np.arange(len(data), dtype=int)[None].T
        self._idata = np.hstack((index, self._data))  # data with index as first column
        self._clusters = []
        self._label_counter = count()
        self._dim_rows = data.shape[0]
        self._dim_cols = data.shape[1]
        self._origin = self._calculate_origin()
        self._cluster_factory = self._get_cluster_factory()

    @abstractmethod
    def _calculate_origin(self):
        """Calculates origin according to algorithm logic"""
        pass

    @abstractmethod
    def _furthest_point_relative_index(self, current_idata):
        """Gets current data with index as first column and calculate furthest point relative index
        i.e. index in current_idata"""
        pass

    @abstractmethod
    def _get_cluster_factory(self):
        """Returns a ClusterFactory to make clusters from label and data with new method"""
        pass

    def _furthest_point_index(self, current_idata):
        """Gets current data with index as first column and calculate furthest point index.
        Returns point index according the initial index (first column of current_idata) and
        relative index in current_idata."""
        relative_index = self._furthest_point_relative_index(current_idata)
        return current_idata[relative_index][0], relative_index

    def _result_array(self):
        """Return result labels array based on accumulated clusters array"""
        result = np.full(fill_value=-1, shape=self._dim_rows)
        for c in range(0, len(self._clusters)):
            cluster = self._clusters[c]
            result[cluster.points_indices] = c
        return np.array(result)

    def __call__(self):
        current_idata = self._idata
        while len(current_idata) > 0:
            current_index = current_idata[:, 0].astype(int)  # this is index of the points in initial data terms
            current_data = current_idata[:, 1:]  # this is current, cropped data without index
            # step 2
            tentative_centroid_index, tentative_centroid_relative_index = self._furthest_point_index(current_idata)
            anomalous_cluster = self._cluster_factory.new(next(self._label_counter), self._data)
            anomalous_cluster.set_points_and_update(np.array([tentative_centroid_index]))
            anomaly = np.full(shape=current_idata.shape, fill_value=False, dtype=bool)
            anomaly[tentative_centroid_relative_index] = True
            while not anomalous_cluster.is_stable:
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
        return self._result_array()
