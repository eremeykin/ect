__author__ = 'eremeykin'
import numpy as np
from ik_means import ik_means
from scipy.spatial import distance as d
import matplotlib.pyplot as plt
import itertools
from cluster import Cluster
from pprint import pprint


p = 0.5
init = False

colors = itertools.cycle(["r", "c", "y", "k"])
markers = itertools.cycle([".", ",", "^", "s"])

def plot(data, clusters):
    plt.axis('equal')
    global init,colors,markers
    if not init:
        for c in clusters:
            setattr(c,'color',next(colors))
            setattr(c,'marker',next(markers))
        init = True
    for c in clusters:
        plt.scatter(data[c.content][:, 0], data[c.content][:, 1], marker=c.marker, color=c.color, s=200)
    plt.pause(p)


def merge(clusters, a, b, distance):
    ca = clusters.pop(a)
    cb = clusters.pop(b)
    new = Cluster(np.hstack((ca.content, cb.content)),
                  (ca.size * ca.centroid + cb.size * cb.centroid) / (ca.size + cb.size))
    setattr(new,'color', ca.color if ca.size>cb.size else cb.color)
    setattr(new,'marker', ca.marker if ca.size>cb.size else cb.marker)
    distance = np.delete(distance,a,0)
    distance = np.delete(distance,a,1)
    distance = np.delete(distance,b,0)
    distance = np.delete(distance,b,1)
    new_column = np.full((distance.shape[0],1),float("inf"))
    distance = np.hstack((distance,new_column))
    new_line = np.full((1,distance.shape[1]),float("inf"))
    distance = np.vstack((distance,new_line))
    cb = new
    Nb = cb.size
    for i in range(len(distance)-1):
        ca = clusters[i]
        Na = ca.size
        distance[i][-1] = ((Na * Nb) / (Na + Nb)) * d.sqeuclidean(ca.centroid, cb.centroid)
        distance[-1][i] = distance[i][-1]
    clusters.append(new)
    return distance


def a_ward(data, clusters, K):
    print(clusters)
    distance = np.full((len(clusters),len(clusters)),float("inf"))
    for a in range(len(clusters)):
            for b in range(a,len(clusters)):
                if a!=b:
                    ca = clusters[a]
                    cb = clusters[b]
                    Na = ca.size
                    Nb = cb.size
                    distance[a][b] = ((Na * Nb) / (Na + Nb)) * d.sqeuclidean(ca.centroid, cb.centroid)
                    distance[b][a] = distance[a][b]
    print("initial:")
    print(distance)
    print("____________")
    # plt.pause(40)
    while len(clusters) > K:
        m = np.argmin(distance)
        min_a = m//len(distance)
        min_b = m%len(distance)

        plot(data, clusters)
        plt.scatter(data[clusters[min_a].content][:, 0], data[clusters[min_a].content][:, 1], marker='o', facecolors='none', s=500,
                    edgecolors='b')
        plt.pause(0.1)
        plt.scatter(data[clusters[min_b].content][:, 0], data[clusters[min_b].content][:, 1], marker='o', facecolors='none', s=500,
                    edgecolors='b')
        plt.pause(0.3)
        plt.clf()
        print(distance)
        distance = merge(clusters, max(min_a, min_b), min(min_a, min_b),distance)
    plot(data,clusters)
    plt.show()
    return clusters


data = np.loadtxt("../tests/data/ikmeans_test2.dat")
l, c = ik_means(data)
# c=[]
# for i,e in enumerate(data):
#     c.append(Cluster(np.array([i]),e))

a_ward(data, c, 3)

