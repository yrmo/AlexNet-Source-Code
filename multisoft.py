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
    B[0, :] = 1
    if fixed >= 0:
        B[0, :fixed+1] = 0
    logBNorms = np.zeros(N+2)
    # Backward pass
    for i in xrange(N - 1, -1, -1):
        B[0,i] = B[0,i+1]
        #B[1, i] = exp(LogBNorms[i+2] + elts[i]) + B[1, i + 1] * (fixed != i)
        for s in xrange(1, size + 1):
            B[s, i] = B[s - 1, i + 1] * exp(elts[i]) + B[s, i + 1] * (fixed != i)
        norm = B[:,i].sum()
        B[:,i] /= norm
        logBNorms[i] = log(norm) + logBNorms[i+1]

    #print LogBNorms
    # Log partition function
    #print B
    #print B * np.exp(LogBNorms)
    logZ = log(B[size, 0]) + logBNorms[0]
    #print logZ; sys.exit()
    F = np.zeros((size + 1,)) # Forward column
    F[0] = 1
    # Forward pass
    # Compute z_j for each j (unnormalized prob)
    z = np.zeros(N)
    logFNorm = 0
    for i in xrange(1, N + 1):
        for s in xrange(size, -1, -1):
            if s < size:
                z[i - 1] += F[s] * B[size - 1 - s, i]
            if s > 0:
                F[s] = F[s - 1] * exp(elts[i - 1]) + F[s] * (fixed != i - 1)
            elif fixed == i - 1:
                F[0] = 0
        norm = F.sum()
        F /= norm
        z[i - 1] *= exp(elts[i - 1] + logBNorms[i] + logFNorm - logZ)
        logFNorm += log(norm)
    return z, 1

# Checks the gradient with respect to the objective
# E = log(y_i)
# where y_i = z_i/Z and i = the index of the correct label
def check_grad(elts, size, correct=0):
    eps = 0.01
    N = len(elts)
    z, Z = sumprod(elts, size)
    cz, CZ = sumprod(elts, size, fixed=correct)
    
    y = z / Z
    Cy = cz / CZ
    grad = Cy - y
    print "Analytic gradient: "
    print grad
    
    grad_num = np.zeros_like(grad)
    for i in xrange(N):
        tmp = elts[i]
        elts[i] += eps
        z, Z = sumprod(elts, size)
        y_n = z / Z
        grad_num[i] = (log(y_n[correct]) - log(y[correct])) / eps
        elts[i] = tmp
    print "Numeric gradient: "
    print grad_num
    
if __name__ == "__main__":
    nr.seed(2)
    N = 5 # The number of outputs in the softmax
    size = 2 # The size of the multisoft set
    fixed = 2 # Force this index to be on (negative = don't)
    elts = nr.randn(N)
    elts -= elts.max()
    print elts
    
    dp_z, dp_Z = sumprod(elts, size, fixed=fixed)
    bf_Z = sumprod_brute(elts, size, fixed=fixed)
    print "Brute force Z: %f" % bf_Z
    print "DP Z: %f" % dp_Z
    
    print "Brute force z/Z:"
    bf_z = np.zeros(N)
    for i in xrange(N):
        for s in xrange(size):
            bf_z[i] += sumprod_brute(elts[:i], s, fixed=fixed) * sumprod_brute(elts[i+1:], size - 1 - s, fixed=fixed-i-1)
        bf_z[i] *= exp(elts[i])

    print bf_z / bf_Z
    
    print "DP z/Z:"
    print dp_z / dp_Z
    
    check_grad(elts, size, correct=1)
    
    
    
