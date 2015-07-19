from sigmoid_func import sigmoid
from softmax import softmax

import numpy as np


def forward_backward_prop(data, labels, params):
    """ Forward and backward propagation for a two-layer sigmoidal network """
    ###################################################################
    # Compute the forward propagation and for the cross entropy cost, #
    # and backward propagation for the gradients for all parameters.  #
    ###################################################################
    
    ### Unpack network parameters (do not modify)
    t = 0
    W1 = np.reshape(params[t:t+dimensions[0]*dimensions[1]], (dimensions[0], dimensions[1]))
    t += dimensions[0]*dimensions[1]
    b1 = np.reshape(params[t:t+dimensions[1]], (1, dimensions[1]))
    t += dimensions[1]
    W2 = np.reshape(params[t:t+dimensions[1]*dimensions[2]], (dimensions[1], dimensions[2]))
    t += dimensions[1]*dimensions[2]
    b2 = np.reshape(params[t:t+dimensions[2]], (1, dimensions[2]))
    
    ### YOUR CODE HERE: forward propagation
    
    # cost = ...
    Z1 = np.dot(W1, data) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W1, A1) + b2
    A2 = softmax(Z2)
    N, D = data.shape
    cost = (-1/N)*np.sum(np.log(A2))

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation

    #gradW1 = ...
    #gradb1 = ...
    #gradW2 = ...
    #gradb2 = ...
    
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad
