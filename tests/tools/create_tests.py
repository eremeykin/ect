__author__ = 'eremeykin'
from generator import generate_norm
import numpy as np

C1 = generate_norm([(0,6),(4,7)], 130)
C2 = generate_norm([(4,10),(8,11)], 150)
C3 = generate_norm([(8,10),(4,7)], 100)
print(np.vstack((C1,C2,C3)))
np.savetxt('data/test1.dat',np.vstack((C1,C2,C3)))
