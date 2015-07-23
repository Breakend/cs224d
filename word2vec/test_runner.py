from sigmoid_func import sigmoid, sigmoid_grad
from softmax import softmax
from grad_check import gradcheck_naive
from simple_nn import forward_backward_prop

import numpy as np

import unittest
import random

class NeuralNetworkTests(unittest.TestCase):

  def test_sigmoid(self):
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    np.testing.assert_allclose(f, np.array([[ 0.73105858, 0.88079708], [ 0.26894142, 0.11920292]]))
    np.testing.assert_allclose(g, np.array([[ 0.19661193, 0.10499359], [ 0.19661193, 0.10499359]]))

  def test_softmax(self):
    np.testing.assert_allclose(softmax(np.array([[1001,1002],[3,4]])), np.array([[ 0.26894142, 0.73105858], [ 0.26894142, 0.73105858]]))
    np.testing.assert_allclose(softmax(np.array([[-1001,-1002]])), np.array([[ 0.73105858, 0.26894142]]))
    np.testing.assert_allclose(softmax(np.array([3,4])), np.array([ 0.26894142, 0.73105858]))

  def test_grad_check(self):
    # Sanity check for the gradient checker
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "=== For autograder ==="
    self.assertTrue(gradcheck_naive(quad, np.array(123.456)))      # scalar test
    self.assertTrue(gradcheck_naive(quad, np.random.randn(3,)))    # 1-D test
    self.assertTrue(gradcheck_naive(quad, np.random.randn(4,5)))   # 2-D test

  def test_simple_nn(self):
    # Set up fake data and parameters for the neural network
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )
    self.assertTrue(gradcheck_naive(lambda params: forward_backward_prop(dimensions, data, labels, params), params))

if __name__ == '__main__':
    unittest.main()
