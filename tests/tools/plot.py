__author__ = 'eremeykin'
import numpy as np
from itertools import cycle

colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k', ])
markers = cycle(['o', 'p', '.', 's', '8', 'h'])
lm_map = dict()


def plot(data, labels=None):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.ion()
    if labels is None:
        labels = np.full(len(data), 1, dtype=int)
    for l in np.unique(labels):
        c = lm_map.get(l, (next(colors), 'w'))[0]
        m = lm_map.get(l, ('o', next(markers)))[1]
        lm_map[l] = (c, m)
        plt.scatter(data[:, 0][labels == l], data[:, 1][labels == l], s=150, marker=m, color=c)
    # plt.show()

    # plt.clf()
    plt.pause(1.5)
    plt.cla()


def hold_plot(data, labels=None):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.ioff()
    if labels is None:
        labels = np.full(len(data), 1, dtype=int)
    for l in np.unique(labels):
        c = lm_map.get(l, (next(colors), 'w'))[0]
        m = lm_map.get(l, ('o', next(markers)))[1]
        lm_map[l] = (c, m)
        plt.scatter(data[:, 0][labels == l], data[:, 1][labels == l], s=150, marker=m, color=c)
    plt.show()


if __name__ == '__main__':
    d = np.loadtxt('../data/ikmeans_test4.dat')
    plot(d)
