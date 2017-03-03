import numpy as np


def generate(n_clusters, dim, zone, sample_size):
    zone = np.abs(zone)
    data = np.empty(dtype=float, shape=(0, dim))
    s = np.random.normal(sample_size / n_clusters, sample_size / (n_clusters * 3), n_clusters)
    print(s)
    s = np.abs(s) / sum(s)
    print(s)
    for cluster in range(0, n_clusters):
        cov = np.random.uniform(low=0, high=zone / n_clusters * dim, size=dim)
        cov = np.diag(cov)
        mean = np.random.uniform(low=-zone, high=zone, size=dim)
        print(s[cluster] * sample_size)
        d = np.random.multivariate_normal(mean, cov, size=int(s[cluster] * sample_size))
        data = np.row_stack((data, d))
    return data


def generate_specific():
    data = np.empty(dtype=float, shape=(0, 2))
    cov = np.random.uniform(low=0, high=10, size=2)
    cov = np.diag(cov)
    d1 = np.random.multivariate_normal(np.array([-5, -7]), cov, size=150)
    data = np.row_stack((data, d1))
    d2 = np.random.multivariate_normal(np.array([-5, 7]), cov, size=150)
    data = np.row_stack((data, d2))
    cnd = np.apply_along_axis(lambda x: np.abs(x[0]) - 5 > 2 > np.abs(x[1]), 1, data)
    data = data[~cnd]
    cov = np.diag([25, 0.01])
    d3 = np.random.multivariate_normal(np.array([0, 0]), cov, size=50)
    data = np.row_stack((data, d3 + np.array([30, 0])))
    for a in np.arange(np.pi / 3,  np.pi - np.pi / 3, np.pi / 3):
        d4 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        data = np.row_stack((data, d3.dot(d4) + np.array([30, 0])))

    return data


if __name__ == "__main__":
    gen_data = generate_specific()
    np.savetxt('../../tests/data/ikmeans_test11.dat', gen_data)
