##
# Miscellaneous helper functions
##

from numpy import *
import random

def random_weight_matrix(m, n):
    #### YOUR CODE HERE ####
    epsilon = sqrt(6.0)/sqrt(m + n)
    A0 = array([[random.uniform(-epsilon, epsilon) for x in range(0,n)] for y in range(0,m)])
    #### END YOUR CODE ####
    assert(A0.shape == (m,n))
    return A0
