import numpy as np


d_1, d_2 = 1, -1


# np.random.seed(235234)


def kovaleva(min_cluster_card, K, size, a):
    N = size[0]
    V = size[1]
    data = np.empty((0, V))
    labels = np.empty((1, 0))
    residue = N - K * min_cluster_card
    if residue < 0:
        raise ValueError('K or min_cluster_car is too big: K*min_cluster_card>N')
    # calculate cardinality for each cluster
    r = np.random.uniform(low=0, high=1, size=K - 1)
    r.sort()
    r1 = np.hstack(([0], r))
    r2 = np.hstack((r, [1]))
    r = r2 - r1
    makeweights = (np.floor(r * residue)).astype(int)
    discrepancy = residue - sum(makeweights)
    makeweights = np.hstack((makeweights[:discrepancy] + 1, makeweights[discrepancy:]))
    for k in range(0, K):
        mean = np.random.uniform(low=a * d_2, high=a * d_1, size=V)
        c = np.random.uniform(low=0.025 * (d_1 - d_2) / 12, high=0.05 * (d_1 - d_2) / 12, size=V)
        cov = np.diag(c)
        cluster_data = np.random.multivariate_normal(mean, cov, min_cluster_card + makeweights[k])
        data = np.row_stack((data, cluster_data))
        labels = np.column_stack((labels, np.full((1, len(cluster_data)), k, int)))
    return data, labels[0]


if __name__ == "__main__":
    i=0
    for n, card, K in [(10, 3, 3), (50, 10, 4), (100, 15, 5), (500, 20, 8), (1000, 50, 18), (2500, 65, 20),
                       (5000, 80, 25), (10000, 90, 30)]:
        i+=1
        gen_data, gen_labels = kovaleva(card, K, (n, K), 0.9)
        np.savetxt("tmp/test{}-points".format(i), gen_data)
        np.savetxt("tmp/test{}-labels".format(i), gen_labels)
    # np.savetxt('../../tests/data/ikmeans_test12.dat', gen_data)
    # to = TestObject()
    # to.plot(gen_data, gen_labels, show_num=False)
    from tests.tools.plot import TestObject
    gen_data, gen_labels = kovaleva(60, 25, (1500, 2), 0.9)
    np.savetxt('../../tests/data/ikmeans_test12.dat', gen_data)
    to = TestObject()
    to.plot(gen_data, gen_labels, show_num=False)
