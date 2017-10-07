import numpy as np


class KDE(object):
    def __init__(self, points):
        self.points = points
        self.n = len(self.points)
        self.h = np.std(self.points) * (4 / (3 * self.n)) ** (1 / 5)

    @staticmethod
    def K(x):  # Gaussian density
        return (2 * np.pi) ** (-0.5) * np.e ** (-0.5 * x ** 2)

    def kde(self, x):
        s_i = np.apply_along_axis(lambda p: KDE.K((x - p) / self.h), axis=0, arr=self.points)
        return (self.n * self.h) ** (-1) * sum(s_i)


class Direction(object):
    def __init__(self, cluster, vector):
        vector = vector / np.linalg.norm(vector)
        Pns = np.dot(cluster.cluster, vector)
        self.sort_indexes = np.argsort(Pns)
        self.sorted_global_indexes = cluster.global_indexes[[self.sort_indexes]]
        self.P = np.sort(Pns)
        kde = np.vectorize(KDE(self.P).kde)
        self.P_kde = kde(self.P)
        self._find_mins()
        self.mins_count = len(self.mins)
        try:
            dmi, dmv = min(self.mins, key=lambda x: x[1])
            if kde((self.P[dmi] + (self.P[dmi+1]-self.P[dmi])/100)) < kde(self.P[dmi]):
                dmi += 1
                dmv = kde((self.P[dmi] + (self.P[dmi+1]-self.P[dmi])/100))
            self.deepest_min_index = dmi
            self.deepest_min_value = dmv
        except ValueError:
            self.deepest_min_index, self.deepest_min_value = None, None

    def _find_mins(self):
        self.mins = []
        for i in range(1, len(self.P_kde) - 1):
            left_value = self.P_kde[i - 1]
            current_value = self.P_kde[i]
            right_value = self.P_kde[i + 1]
            if left_value > current_value and right_value > current_value:
                self.mins.append((i, current_value))
