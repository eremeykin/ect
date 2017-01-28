import numpy as np


def imwk_means(data, centroids):
    def dist_to_p_beta(c, p, beta, w):
        return lambda y: np.dot((np.abs(y - c) ** p), w ** beta)

    K = len(centroids)
    yi_to_ck = np.apply_along_axis(dist_to_p_beta(), 1, data)
    pass


if __name__ == "__main__":
    data = np.loadtxt("../tests/data/ikmeans_test4.dat")
    print(data)
