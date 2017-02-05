__author__ = 'eremeykin'
from generator import generate_norm
import numpy as np
import random
from tests.tools.plot import hold_plot, plot

Ci = np.array([]).reshape(0, 2)


class Cluster:
    def __init__(self, minx, maxx, miny, maxy, count):
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.count = count


clusters = [Cluster(10, 15, 30, 31, 20), Cluster(15, 18, 24, 27, 30), Cluster(9, 12, 22, 24, 10)]
for c in clusters:
    Ci_new = generate_norm([(c.minx, c.maxx), (c.miny, c.maxy)], c.count)
    Ci = np.vstack((Ci, Ci_new))
hold_plot(Ci)
np.savetxt('../data/ikmeans_test6.dat', Ci)
