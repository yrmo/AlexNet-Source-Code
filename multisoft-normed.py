import sys
import numpy as np
import numpy.random as nr
from math import exp, log

def sumprod_brute(elts, size, fixed = -1):
    if size > len(elts):
        return 0
    if fixed >= 0 and fixed < len(elts):
        if size == 0:
            return 0
        z = 0
        for s in xrange(size):
            z += sumprod_brute(elts[:fixed], s) * sumprod_brute(elts[fixed+1:], size - 1 - s)
        return z * exp(elts[fixed])
    if size == 0:
        return 1
    return exp(elts[0]) * sumprod_brute(elts[1:], size - 1) + sumprod_brute(elts[1:], size)

# Returns sum over all subsets of given size of product
# of exp of elements.
# Also returns, for each index, the said sum given that the element
# at that index is in the subset.
def sumprod(elts, size, fixed = -1):
    N = len(elts)
    B = np.zeros((size + 1, N + 1)) # Backward lattice
    B[0, N] = 1

    logBNorms = np.zeros(N + 1)
    # Backward pass
    for i in xrange(N - 1, -1, -1):
        B[0, i] = (i >= size and i > fixed) * B[0, i + 1] # This can get quite small
        for s in xrange(max(1, size - i), size + 1):
            B[s, i] = B[s - 1, i + 1] * exp(elts[i]) + B[s, i + 1] * (fixed != i)
        norm = B[:,i].sum()
        B[:,i] /= norm
        logBNorms[i] = log(norm) + logBNorms[i + 1]

    F = np.zeros((size + 1,)) # Forward column
    F[0] = 1

    # Forward pass
    # Compute y_j for each j (marginal prob)
    y = np.zeros(N)
    logFNorm = -logBNorms[0] # Subtract log partition function 
    for i in xrange(1, N + 1):
        for s in xrange(size, -1, -1):
            if s < size:
                y[i - 1] += F[s] * B[size - 1 - s, i]
            if s > 0:
                F[s] = F[s - 1] * exp(elts[i - 1]) + F[s] * (fixed != i - 1)
            elif fixed == i - 1:
                F[0] = 0
        norm = F.sum()
        F /= norm
        y[i - 1] *= exp(elts[i - 1] + logBNorms[i] + logFNorm)
        logFNorm += log(norm)
    return y

# Computes log(exp(x) + exp(y))
def logadd(x, y):
    if x == -np.inf and y == -np.inf:
        return -np.inf
    M = max(x,y)
    m = min(x,y)
    diff = M - m
    return M if diff >= 15 else M + log(1 + exp(-diff))
    

# Returns sum over all subsets of given size of product
# of exp of elements.
# Also returns, for each index, the said sum given that the element
# at that index is in the subset.
def sumprod_logspace(elts, size, fixed = -1):
    N = len(elts)
    logB = -np.inf * np.ones((size + 1, N + 1)) # Backward lattice

    logB[0, :] = 0
    if fixed >= 0:
        logB[0, :fixed + 1] = -np.inf

    # Backward pass
    for i in xrange(N - 1, -1, -1):
        for s in xrange(max(1, size - i), size + 1):
            logB[s, i] = logadd(logB[s - 1, i + 1] + elts[i], logB[s, i + 1] if fixed != i else -np.inf)

    logF = -np.inf * np.ones((size + 1,)) # Forward column
    logF[0] = 0

    # Forward pass
    # Compute y_j for each j (marginal prob)
    logy = -np.inf * np.ones(N)
    logFNorm = -logB[size, 0] # Subtract log partition function 
    for i in xrange(1, N + 1):
        for s in xrange(size, -1, -1):
            if s < size:
                logy[i - 1] = logadd(logy[i - 1], logF[s] + logB[size - 1 - s, i])
            if s > 0:
                logF[s] = logadd(logF[s - 1] + elts[i - 1], logF[s] if fixed != i - 1 else -np.inf)
            elif fixed == i - 1:
                logF[0] = -np.inf

        logy[i - 1] += elts[i - 1] + logFNorm
    return np.exp(logy)

# Checks the gradient with respect to the objective
# E = log(y_i)
# where y_i = z_i/Z and i = the index of the correct label
def check_grad(elts, size, correct=0):
    eps = 0.01
    N = len(elts)
    y = sumprod_logspace(elts, size)
    Cy = sumprod_logspace(elts, size, fixed=correct)
    
    grad = Cy - y
    print "Analytic gradient: "
    print grad
    
    grad_num = np.zeros_like(grad)
    for i in xrange(N):
        tmp = elts[i]
        elts[i] += eps
        y_n = sumprod_logspace(elts, size)
        grad_num[i] = (log(y_n[correct]) - log(y[correct])) / eps
        elts[i] = tmp
    print "Numeric gradient: "
    print grad_num
    
if __name__ == "__main__":
    nr.seed(2)
    N = 5 # The number of outputs in the softmax
    size = 2 # The size of the multisoft set
    fixed = -2 # Force this index to be on (negative = don't)
    elts = nr.randn(N)
    elts -= elts.max()
    elts = np.array([-0.071459650993347, -0.517264485359192, -0.128548145294189, -0.113207340240479 ,0.000000000000000])
    print elts
    
    dp_y = sumprod_logspace(elts, size, fixed=fixed)
    bf_Z = sumprod_brute(elts, size, fixed=fixed)
    print "Brute force Z: %f" % bf_Z
    
    print "Brute force z/Z:"
    bf_z = np.zeros(N)
    for i in xrange(N):
        for s in xrange(size):
            bf_z[i] += sumprod_brute(elts[:i], s, fixed=fixed) * sumprod_brute(elts[i+1:], size - 1 - s, fixed=fixed-i-1)
        bf_z[i] *= exp(elts[i])

    print bf_z / bf_Z
    
    print "DP z/Z:"
    print dp_y
    
    check_grad(elts, size, correct=3)