from clustering.cluster_structure import ClusterStructure
from abc import ABC, abstractmethod


class AgglomerativeClusterStructure(ClusterStructure, ABC):
    @abstractmethod
    def dist_point_to_point(self, point1, point2, cluster_of_point1=None):
        pass

    @abstractmethod
    def dist_point_to_cluster(self, point, cluster):
        pass

    @abstractmethod
    def dist_cluster_to_cluster(self, cluster1, cluster2):
        pass
