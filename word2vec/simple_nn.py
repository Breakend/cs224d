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
    # labels is (20, 10) (20 1-hot vectors) - this is y
    # data is (20, 10) - this is x
    # W1 is (10, 5)
    # W2 is (5, 10)
    # b1 is (1, 5)
    # b2 is (1, 10)

    a = data.dot(W1) + b1
    h = sigmoid(a) # hidden layer
    y_hat = softmax(h.dot(W2) + b2) # Top classifier layer
    N, D = data.shape
    (Dx, H) = W1.shape

    # TODO: may need to change this to sum over rows and then sum up rows?
    # cost = np.sum(-np.sum(np.multiply(labels, np.log(y_hat)), axis=1).reshape((N, 1)))
    cost_per_datapoint = -np.sum(labels * np.log(y_hat), axis=1).reshape((N, 1)) # sum over rows
    cost = np.sum(cost_per_datapoint)

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    #gradW1 = ...
    #gradb1 = ...
    #gradW2 = ...
    #gradb2 = ...

    # d_y_hat/d_W2
    J_theta = y_hat - labels
    # theta_W2 = h
    # theta_h = W2
    h_a = h * (1.0 - h)
    a_W1 = data
    y_hathw = J_theta.dot(W2.T)*h_a

    gradW2 = h.T.dot(J_theta)
    gradW1 = data.T.dot(y_hathw)
    # gradW1 = np.dot(data.T, np.dot(J_theta, theta_h.T) * h_a)
    gradb1 = np.sum(y_hathw, axis=0).reshape((1, H))
    gradb2 = np.sum(J_theta, axis=0).reshape((1, D))


    assert gradW1.shape == W1.shape
    assert gradb1.shape == b1.shape
    assert W2.shape == gradW2.shape
    assert gradb2.shape == b2.shape

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad
