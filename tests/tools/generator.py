__author__ = 'eremeykin'
import numpy as np

def generate_norm(minmax, N):
    result = np.empty((N,0))
    for min,max in minmax:
        r = max - min
        mu = min+r/2
        sigma = (max-mu)/1.96
        current = np.random.normal(mu, sigma, (N,1))
        result = np.hstack((result,current))
    return result
