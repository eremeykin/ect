from clustering.divisive.depddp import DEPDDP
from clustering.divisive.utils.bikm_r_cluster_structure import BiKMeansRClusterStructure


class BiKMeansR(DEPDDP):
    def __init__(self, data, epsilon, directions_num=None, second_chance=False, split_by='ik_means'):
        self._epsilon = epsilon
        self._directions_num = directions_num
        self._second_chance = second_chance
        self._split_by = split_by
        super().__init__(data)

    def _create_cluster_structure(self, data):
        return BiKMeansRClusterStructure(data, self._epsilon,
                                         self._directions_num,
                                         self._second_chance,
                                         self._split_by)
