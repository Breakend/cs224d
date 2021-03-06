import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4
    h2 = (2.0*h)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### YOUR CODE HERE: try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it
        ### possible to test cost functions with built in randomness later
        # random.setstate(rndstate)
        # numgrad = (f(x[ix]+h)[0] - f(x[ix]-h)[0])/h2
        random.setstate(rndstate)
        x[ix] += h
        f_hi = f(x)[0]
        random.setstate(rndstate)
        x[ix] -= 2.0 * h
        f_low = f(x)[0]
        numgrad = (f_hi - f_low) / (2.0 * h)
        x[ix] += h

        ### END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return False

        it.iternext() # Step to next dimension

    print "Grad check passed!"
    return True
