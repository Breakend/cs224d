from sigmoid_func import sigmoid, sigmoid_grad
from softmax import softmax
from grad_check import gradcheck_naive
from simple_nn import forward_backward_prop
from word2vec import *

import numpy as np

import unittest
import random

class NeuralNetworkTests(unittest.TestCase):

  def test_sigmoid(self):
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    np.testing.assert_array_almost_equal(f, np.array([[ 0.73105858, 0.88079708], [ 0.26894142, 0.11920292]]))
    np.testing.assert_array_almost_equal(g, np.array([[ 0.19661193, 0.10499359], [ 0.19661193, 0.10499359]]))

  def test_softmax(self):
    np.testing.assert_array_almost_equal(softmax(np.array([[1001,1002],[3,4]])), np.array([[ 0.26894142, 0.73105858], [ 0.26894142, 0.73105858]]))
    np.testing.assert_array_almost_equal(softmax(np.array([[-1001,-1002]])), np.array([[ 0.73105858, 0.26894142]]))
    np.testing.assert_array_almost_equal(softmax(np.array([3,4])), np.array([ 0.26894142, 0.73105858]))

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

  def test_word2vec(self):
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = self.dummySampleTokenIdx
    dataset.getRandomContext = self.getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== For autograder ==="
    cost, grad, pred = skipgram(dataset, "c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    grad_val1 = np.array([[ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [-1.26947339, -1.36873189,  2.45158957],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]])
    pred_val1 = np.array([[-0.41045956,  0.18834851,  1.43272264],
        [ 0.38202831, -0.17530219, -1.33348241],
        [ 0.07009355, -0.03216399, -0.24466386],
        [ 0.09472154, -0.04346509, -0.33062865],
        [-0.13638384,  0.06258276,  0.47605228]])
    np.testing.assert_array_almost_equal(cost, np.array([ 11.166109]))
    np.testing.assert_array_almost_equal(grad, grad_val1)
    np.testing.assert_array_almost_equal(pred, pred_val1)

    cost_val2 = np.array([ 14.09527265])
    grad_val2 = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [-3.40325278, -2.74731195, -0.95360761],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ]])
    pred_val2 = np.array([[-0.49853822,  0.22876535,  1.74016407],
       [-0.22716495,  0.10423969,  0.79292674],
       [-0.22764219,  0.10445868,  0.79459256],
       [-0.94807832,  0.43504684,  3.30929863],
       [-0.32248118,  0.14797767,  1.1256312 ]])
    cost, grad, pred = skipgram(dataset, "c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
    np.testing.assert_array_almost_equal(grad, grad_val2)
    np.testing.assert_array_almost_equal(pred, pred_val2)
    np.testing.assert_array_almost_equal(cost, cost_val2)
    # print cbow(dataset, "a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    # print cbow(dataset, "a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)

  def test_normalize_rows(self):
      np.testing.assert_array_almost_equal(normalizeRows(np.array([[3.0,4.0],[1, 2]])), np.array([[0.6, 0.8], [0.4472, 0.8944]]))

  def dummySampleTokenIdx(self):
      return random.randint(0, 4)

  def getRandomContext(self, C):
      tokens = ["a", "b", "c", "d", "e"]
      return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]

if __name__ == '__main__':
    unittest.main()
