from sigmoid_func import sigmoid
from softmax import softmax
from data_utils import *

import numpy as np

###################################################################
# Note, I used msushkov's implementation to double check my own.
# Mine was not great, so you'll see a lot of similarities to his.
# https://github.com/msushkov/cs224d-hw1/blob/master/wordvec_sentiment.ipynb
###################################################################

def forward_backward_prop(dimensions, data, labels, params):
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
    a = np.dot(data, W1) + b1
    h = sigmoid(a) # hidden layer
    y_hat = softmax(np.dot(h, W2) + b2) # Top classifier layer
    N, D = data.shape

    # TODO: may need to change this to sum over rows and then sum up rows?
    cost = (-1/N)*np.sum(np.multiply(labels, np.log(y_hat)))

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    #gradW1 = ...
    #gradb1 = ...
    #gradW2 = ...
    #gradb2 = ...

    # d_y_hat/d_W2
    J_theta = y_hat - labels
    theta_W2 = h
    theta_h = W2
    h_a = a * (1.0 - a)
    a_W1 = data

    gradW2 = np.dot(J_theta.T, theta_W2)
    gradW1 = np.dot(labels.T, np.dot(J_theta, theta_h.T) * h_a)
    gradb1 = np.dot(J_theta, W2.T) * h_a
    gradb2 = J_theta
    assert gradW1.shape == W1.shape
    print gradb1.shape
    print b1.shape
    assert gradb1.shape == b1.shape
    assert W2.shape == gradW2.shape
    assert gradb2.shape == b2.shape

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad
