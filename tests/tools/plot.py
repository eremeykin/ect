__author__ = 'eremeykin'
import numpy as np
from itertools import cycle
import matplotlib
from time import gmtime, strftime
import os
import shutil

# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

clist = ['b', 'g', 'r', 'c', 'm', 'y', 'k', ]
colors = cycle(clist)
markers = cycle(['o', 'p', '.', 's', '8', 'h'])
size = cycle([75, 150, 125, 100])
lm_map = dict()


class TestObject(object):
    LOG_PATH = "../tests/test_logs"
    if os.path.exists(LOG_PATH):
        shutil.rmtree(LOG_PATH)
    os.makedirs(LOG_PATH)

    def __init__(self, test_name=None):
        self.test_name = test_name
        self.counter = 0
        self.axes = None
        self.x_lim = None
        self.y_lim = None

    def set_lims(self, xlim, ylim):
        if self.x_lim is None:
            self.x_lim = xlim
        if self.y_lim is None:
            self.y_lim = ylim

    def plot(self, data, labels=None, centroids=None, show_num=True, prefix=''):
        fig = plt.figure()
        ax = fig.add_subplot(111)


        plt.axis('equal')
        if self.x_lim is not None: plt.xlim(self.x_lim)
        if self.y_lim is not None: plt.ylim(self.y_lim)

        # ax1 = np.arange(0, max(data[:, 0]), 1)
        # ax2 = np.arange(min(data[:, 0]), 0, 1)
        # xticks = np.concatenate((ax1, ax2), axis=0)
        # ay1 = np.arange(0, max(data[:, 1]), 1)
        # ay2 = np.arange(min(data[:, 1]), 0, 1)
        # yticks = np.concatenate((ay1, ay2), axis=0)
        # ax.set_xticks(xticks)
        # ax.set_yticks(yticks)

        # plt.ion()
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
        ind = 0
        if show_num:
            for i, j in data:
                label = labels[ind]
                ind += 1
                ax.annotate(str(label), xy=(i, j), xytext=(4, 3), textcoords='offset points')
        plt.grid(True)
        plt.savefig(
            '{}/{}  - {} #{} [{}].png'.format(TestObject.LOG_PATH, self.test_name, prefix, self.counter,
                                              strftime(" %H:%M (%d.%m)", gmtime()))
        )

        self.set_lims(plt.axes().get_xlim(), plt.axes().get_ylim())
        self.counter += 1

    def hold_plot(self, data, labels=None):
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


class BlankTestObject(TestObject):
    def __init__(self):
        super().__init__()

    def plot(self, *args):
        pass

    def hold_plot(self, *args):
        pass


if __name__ == '__main__':
    d = np.loadtxt('../data/ikmeans_test4.dat')
    plot(d)
