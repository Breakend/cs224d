from sigmoid_func import sigmoid
from softmax import softmax

import numpy as np

###################################################################
# Note, I used msushkov's implementation to double check my own.
# Mine was not great, so you'll see a lot of similarities to his.
# https://github.com/msushkov/cs224d-hw1/blob/master/wordvec_sentiment.ipynb
###################################################################

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
    h = sigmoid(np.dot(W1, data) + b1) # hidden layer
    y_hat = softmax(np.dot(W2, h) + b2) # Top classifier layer
    N, D = data.shape

    # TODO: may need to change this to sum over rows and then sum up rows?
    cost = (-1/N)*np.sum(np.dot(labels, np.log(A2)))

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation

    #gradW1 = ...
    #gradb1 = ...
    #gradW2 = ...
    #gradb2 = ...

    # d_y_hat/d_W2
    gradW2 = np.dot((y_hat - labels), h)
    gradW1 = np.do
    assert gradW1.shape == W1.shape
    assert gradb1.shape == b1.shape
    assert W2.shape == gradW2.shape
    assert gradb2.shape == b2.shape

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))

    return cost, grad
