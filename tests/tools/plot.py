__author__ = 'eremeykin'
import numpy as np
from itertools import cycle

clist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', ]
colors = cycle(clist)
markers = cycle(['o', 'p', '.', 's', '8', 'h'])
size = cycle([75, 100, 125, 150])
lm_map = dict()


def plot(data, labels=None, centroids=None):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.axis('equal')
    plt.ion()
    if labels is None:
        labels = np.full(len(data), 1, dtype=int)
    for l in np.unique(labels):
        c = lm_map.get(l, (next(colors), 'b', 50))[0]
        m = lm_map.get(l, ('o', next(markers), 50))[1]
        s = lm_map.get(l, ('o', 'b', 50))[2]
        lm_map[l] = (c, m, s)
        plt.scatter(data[:, 0][labels == l], data[:, 1][labels == l], s=s, marker=m, color=c)
    if not centroids is None:
        for ct in centroids:
            plt.scatter(ct[0], ct[1], s=175, marker='o', facecolors='none', edgecolors='r')
            plt.scatter(ct[0], ct[1], s=200, marker='x', color='r')
    plt.show()
    plt.pause(15)


def hold_plot(data, labels=None):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.axis('equal')
    if labels is None:
        labels = np.full(len(data), 1, dtype=int)
    for l in np.unique(labels):
        c = lm_map.get(l, (next(colors), 'b', 50))[0]
        m = lm_map.get(l, ('o', next(markers), 50))[1]
        s = lm_map.get(l, ('o', 'b', next(size)))[2]
        lm_map[l] = (c, m, s)
        plt.scatter(data[:, 0][labels == l], data[:, 1][labels == l], s=s, marker=m,
                    color=c)
    plt.show()


if __name__ == '__main__':
    d = np.loadtxt('../data/ikmeans_test4.dat')
    plot(d)
