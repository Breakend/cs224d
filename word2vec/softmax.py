import random
import numpy as np
from data_utils import *
import matplotlib.pyplot as plt


def softmax(x):
    """ Softmax function """
    ###################################################################
    # Compute the softmax function for the input here.                #
    # It is crucial that this function is optimized for speed because #
    # it will be used frequently in later code.                       #
    # You might find numpy functions np.exp, np.sum, np.reshape,      #
    # np.max, and numpy broadcasting useful for this task. (numpy     #
    # broadcasting documentation:                                     #
    # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)  #
    # You should also make sure that your code works for one          #
    # dimensional inputs (treat the vector as a row), you might find  #
    # it helpful for your later problems.                             #
    ###################################################################

    ### YOUR CODE HERE
    # If the input is a one dimensional vector, reshape such that it's a 1 x D
    # dimensional vector
    input_x = x
    if len(x.shape) == 1:
        x = x.reshape((1, x.shape[0]))

    # Subtract maximum to normalize for numerical stability
    x = x - np.max(x, axis=1)[:,np.newaxis]
    # Exponential
    result = np.exp(x)
    # Sum
    factor = np.sum(result, axis=1)
    result = result / factor[:, np.newaxis]
    result = result.reshape(input_x.shape)
    ### END YOUR CODE

    return result
