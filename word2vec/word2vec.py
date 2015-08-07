# Implement your skip-gram and CBOW models here
from softmax import softmax
from sigmoid_func import sigmoid

import numpy as np
import random


def softmaxCostAndGradient(dataset, predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector (\hat{r} in #
    #           the written component)                                #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       #
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    # Keep track of outputVector shape
    N, D = outputVectors.shape
    x = predicted.reshape(D, 1)
    Wx = (outputVectors.dot(x)).reshape(1, N)
    theta = softmax(Wx)
    theta_vec = theta.reshape((N,))
    cost = -np.log(theta_vec[target])

    gradPred = theta.dot(outputVectors) - outputVectors[target].reshape((1, D))
    gradPred = gradPred.reshape((D,))

    grad = theta.T.dot(predicted.reshape((1, D)))
    m = np.zeros(grad.shape)
    m[target, :] = predicted.reshape((1, D))
    grad = grad - m

    ### END YOUR CODE

    assert grad.shape == outputVectors.shape
    assert gradPred.shape == predicted.shape

    return cost, gradPred, grad

def negSamplingCostAndGradient(dataset, predicted, target, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    samples = []
    sample_indices = []
    for i in range(0,10):
        index = dataset.sampleTokenIdx()
        samples.append(outputVectors[index])
        sample_indices.append(index)

    samples = np.array(samples)
    N, D = outputVectors.shape

    # cost = - log(\sigma(v^{i-C+j} \dot h)) + \sum_{k=1}^{K} log(\sigma(v^{(k)} \dot h))
    samples_dot_predicted = sigmoid(samples.dot(predicted.reshape((D,1))))
    predicted_dot_target = sigmoid(predicted.dot(outputVectors[target]))
    cost = -1.0 * np.sum(np.log(samples_dot_predicted)) - np.log(predicted_dot_target)

    # Derivative w.r.t. predicted (h)
    # -(\frac{1}{\sigma(v^{i-C+j} \dot h)})(\grad sigmoid(v^{i-C+j} \dot (h)))(v^{i-C+j})
    # => - (1-sigmoid(samples \dot (predicted)))(samples)
    # + (1-sigmoid(sampled \dot predicted))()
    # import pdb;pdb.set_trace()
    # import pdb; pdb.set_trace()

    sig = predicted_dot_target - 1.0
    gradPred = sig * outputVectors[target].reshape(1, D) + (1.0 - samples_dot_predicted).reshape(1, K).dot(samples)
    gradPred = gradPred.reshape(D,)

    grad = np.zeros(outputVectors.shape)
    grad[target, :] = predicted * sig
    for sample, k in zip(samples, sample_indices):
        grad[k, :] += -1.0 * predicted * (sigmoid(-1.0 * predicted.dot(sample)) - 1.0)

    ### END YOUR CODE


    assert grad.shape == outputVectors.shape
    assert gradPred.shape == predicted.shape

    return cost, gradPred, grad

def skipgram(dataset, currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of no more than 2*C strings, the context #
    #             words                                               #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    # TODO: change this
    N, D = inputVectors.shape

    curr_index = tokens[currentWord]
    curr_vector = inputVectors[curr_index]

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for context_word in contextWords:
        pred_index = tokens[context_word]
        cost_curr, grad_in_curr, grad_out_curr = word2vecCostAndGradient(dataset, curr_vector, pred_index, outputVectors)
        cost += cost_curr
        gradIn[curr_index, :] += grad_in_curr
        gradOut += grad_out_curr

    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(dataset, currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """
    ###################################################################
    # Implement the continuous bag-of-words model in this function.   #
    # Input/Output specifications: same as the skip-gram model        #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    # TODO: this
    cost = 0.0
    gradIn = 0.0
    gradOut = 0.0

    ### END YOUR CODE

    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(dataset, centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """

    ### YOUR CODE HERE
    # TODO: fix this
    (r, c) = x.shape
    row_sums = np.sum(x**2, axis=1)
    x = x / np.sqrt(row_sums.reshape((r, 1)))
    ### END YOUR CODE

    return x
