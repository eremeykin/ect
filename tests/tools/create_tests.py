__author__ = 'eremeykin'
from generator import generate_norm
import numpy as np
import random

N = 15
size_min = 30
size_max = 800
center_minx = 0
center_miny = 0
center_maxx = 100
center_maxy = 100
rangex_max = 20
rangex_min = 3
rangey_max = 20
rangey_min = 3
size = random.randint(size_min, size_max)
center_x = random.randint(center_minx, center_maxx)
center_y = random.randint(center_miny, center_maxy)
rangex = random.randint(rangex_min, rangex_max)
rangey = random.randint(rangey_min, rangey_max)
Ci = generate_norm([(center_x - rangex, center_x + rangex), (center_y - rangey, center_y + rangey)], size)

for i in range(N):
    size = random.randint(size_min, size_max)
    center_x = random.randint(center_minx, center_maxx)
    center_y = random.randint(center_miny, center_maxy)
    rangex = random.randint(rangex_min, rangex_max)
    rangey = random.randint(rangey_min, rangey_max)
    Ci_new = generate_norm([(center_x - rangex, center_x + rangex), (center_y - rangey, center_y + rangey)], size)
    Ci= np.vstack((Ci,Ci_new))
# C1 = generate_norm([(0, 6), (4, 7)], 130)
# C2 = generate_norm([(4, 10), (8, 11)], 150)
# C3 = generate_norm([(8, 10), (4, 7)], 100)
# C4 = generate_norm([(3, 4), (7, 15)], 200)
# C5 = generate_norm([(8, 12), (4, 11)], 80)
# print(np.vstack((C1,C2,C3)))
np.savetxt('../data/ikmeans_test5.dat', Ci)
