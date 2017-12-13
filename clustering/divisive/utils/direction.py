import numpy as np


class KDE:
    def __init__(self, points):
        self.points = points
        self.n = len(self.points)
        self.h = np.std(self.points) * (4 / (3 * self.n)) ** (1 / 5)

    @staticmethod
    def K(x):  # Gaussian density kernel
        return (2 * np.pi) ** (-0.5) * np.e ** (-0.5 * x ** 2)

    def __call__(self, x):
        s_i = np.apply_along_axis(lambda point: KDE.K((x - point) / self.h), axis=0, arr=self.points)
        return (self.n * self.h) ** (-1) * sum(s_i)


class Direction:
    def __init__(self, data, vector, do_norm=False):
        # normalize cluster points
        if do_norm:
            mean = np.mean(data, axis=0)
            range_ = np.max(data, axis=0) - np.min(data, axis=0)
            self._norm_points = (data - mean) / range_
        # convert to unit vector
        vector = vector / np.linalg.norm(vector)
        assert np.abs(np.linalg.norm(vector) - 1) < 0.0001
        # project cluster points to the unit vector
        vector_projections = self._cluster_points @ vector
        # reorder _points_indices so that projections to the vector are ordered
        projections_sort_indices = np.argsort(vector_projections)
        vector_projections = vector_projections[projections_sort_indices]
        self._points_indices = self._points_indices[projections_sort_indices]
        self._cluster_points = self._cluster_points[projections_sort_indices]
        kde = KDE(vector_projections)
        kde_funct = np.vectorize(kde)
        kde_values = kde_funct(vector_projections)
        minima = Direction._find_minima(kde_values)
        try:
            dmi, dmv = min(minima, key=lambda x: x[1])
            # here we check the only shift +1, because we assign deepest
            # minimum point to the right cluster anyway
            small_shift = (vector_projections[dmi + 1] - vector_projections[dmi]) / 100  # why exactly 100?
            if kde(vector_projections[dmi] + small_shift) < kde(vector_projections[dmi]):
                dmi += 1
                dmv = kde(vector_projections[dmi] + small_shift)
            self.deepest_min_index = dmi
            self.deepest_min_value = dmv
        except ValueError:
            self.deepest_min_index, self.deepest_min_value = None, None

    @staticmethod
    def _find_minima(sorted_kde):
        minima = []
        for i in range(1, len(sorted_kde) - 1):
            left_value = sorted_kde[i - 1]
            current_value = sorted_kde[i]
            right_value = sorted_kde[i + 1]
            if left_value > current_value and right_value > current_value:
                minima.append((i, current_value))
        return minima
