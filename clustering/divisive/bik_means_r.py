from clustering.cluster_structure import ClusterStructure


class BiKMeansRClusterStructure(ClusterStructure):
    class Cluster(ClusterStructure.Cluster):
        def __init__(self, cluster_structure, points_indices, with_setup=True):
            super().__init__(cluster_structure, points_indices, with_setup)

    def __init__(self, second_chance=False):
        self._second_chance = second_chance

    def find_best_cluster(self):
        for cluster in self._clusters:
            pass


    def split(self, cluster):
        pass
