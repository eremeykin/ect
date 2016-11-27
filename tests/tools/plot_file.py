__author__ = 'eremeykin'
import  matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data/test1.dat')
print('data=')
print(data[:,0])
C1x = data[:,0][:130]
C1y = data[:,1][:130]

C2x = data[:,0][130:130+150]
C2y = data[:,1][130:130+150]

C3x = data[:,0][130+150:]
C3y = data[:,1][130+150:]


plt.plot(C1x,C1y,'o',color='red')
plt.plot(C2x,C2y,'o',color='blue')
plt.plot(C3x,C3y,'o',color='yellow')
plt.show()