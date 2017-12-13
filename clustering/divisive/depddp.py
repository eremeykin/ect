from clustering.divisive.utils.depddp_cluster_strucutre import DEPDDPClusterStructure
import numpy as np

class DEPDDP:
    def __init__(self, data):
        self._data = data
        self._cluster_structure = DEPDDPClusterStructure(data)
        giant = self._cluster_structure.release_new_cluster(np.arange(len(data), dtype=int))
        self._cluster_structure.add_cluster(giant)
        self._completed = False

    @property
    def cluster_structure(self):
        if not self._completed:
            raise BaseException("Not completed yet")
        return self._cluster_structure

    def __call__(self):
        while True:
            best_cluster = self._cluster_structure.find_best_cluster()
            if best_cluster is None:
                break
            self._cluster_structure.split(best_cluster)
        self._completed = True
        return self._cluster_structure.current_labels()
