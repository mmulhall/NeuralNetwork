import numpy as np
import math, random
from ctypes.wintypes import DOUBLE

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
# input dataset
X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
 
# output dataset            
y = np.array([[0,0,1,1]]).T
 
# seed random numbers to make calculation deterministic
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for item in range(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)
    print("{0},{1},{2},{3},{4},{5}".format(l1, syn0, l1_error[0], l1_error[1], l1_error[2], l1_error[3]))

print("Output After Training:")
print(l1)
