import numpy as np
import logging
from time import clock

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
DELTA = np.finfo(float).eps ** 0.333

def jacobian(func, x, *args, **kwargs):
    """
    estimate Jacobian
    """
    nargs = x.shape[0]  # degrees of freedom
    nobs = x.size / nargs  # number of observations
    LOGGER.debug("degrees of freedom: %d", nargs)
    LOGGER.debug("number of observations: %d", nobs )
    f = lambda x_: func(x_, *args, **kwargs)
    j = None  # matrix of zeros
    for n in xrange(nargs):
        d = np.zeros((nargs, nobs))
        d[n] += x[n] * DELTA
        dx = 2.0 * DELTA * x[n]
        df = (f(x + d) - f(x - d)) / dx
        # derivatives df/d_n
        if j is None:
            j = np.zeros((df.shape[0], nargs, nobs))
        j[n] = df
    return j.T


def jacobs(j):
    nobs, nf, nx = j.shape
    nrows, ncols = nf*nobs, nx*nobs
    jj = np.zeros((nrows, ncols))
    for n in xrange(nobs):
        r, c = n*nf, n*nx
        jj[r:r+nf, c:c+nx] = j[n]
    return jj



if __name__ == "__main__":
    g = lambda x: np.array([x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2,
                            x[1] ** 2 - x[0] ** 2])
    h = lambda x: np.array([(2 * (x[0] + x[1]), 2 * (x[1] + x[0])),
                            (-2 * x[0], 2 * x[1])])
    z = np.array([range(1, 6), range(6, 11)])
    print g(z)
    print "==================================================================="
    print h(z)
    print "==================================================================="
    c = clock()
    j = jacobian(g, z)
    print "elapsed time = %g [s]" % (clock() - c)
    print j
    print j.shape
    print "==================================================================="
    xunc = 0.3
    cov = np.diag(np.repeat(xunc ** 2, 2))  # covariance
    sig = np.sqrt(np.dot(np.dot(j[0], cov), j[0].T))
    print "signma = \n%r" % sig
    std = [np.linalg.norm(j[0, 0] * xunc), np.linalg.norm(j[0, 1] * xunc)]
    print "std-dev = %r" % std
    print "==================================================================="
    jj = jacobs(j)
    print jj
    print "==================================================================="
