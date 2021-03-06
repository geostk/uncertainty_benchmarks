"""
Jacobian estimation.

SunPower Corp. (c) 2016
Mark Mikofski
Confidential and Proprietary - Do Not Distribute
"""

import numpy as np
from time import clock

DELTA = np.finfo(float).eps ** (1.0 / 3.0)


def jacobian(func, x, *args, **kwargs):
    """
    Estimate Jacobian matrices :math:`\frac{\partial f_i}{\partial x_jk}` where
    :math:`k` are independent observations of :math:`x`.

    The independent variable, :math:`x`, must be a numpy array with exactly 2
    dimensions. The first dimension is the number of independent arguments,
    and the second dimentions is the number of observations.

    The function must return a numpy array with exactly 2 dimensions. The first
    is the number of returns and the second dimension corresponds to the number
    of observations.

    Use ``numpy.atleast_2d()`` to get the correct dimensions for scalars.

    :param func: function
    :param x: independent variables grouped by observation
    :return: Jacobian matrices for each observation
    """
    nargs = x.shape[0]  # degrees of freedom
    nobs = x.size / nargs  # number of observations
    f = lambda x_: func(x_, *args, **kwargs)
    j = None  # matrix of zeros
    for n in xrange(nargs):
        dx = np.zeros((nargs, nobs))
        dx[n] += x[n] * DELTA
        df = (f(x + dx) - f(x - dx)) / dx[n] / 2.0
        # derivatives df/d_n
        if j is None:
            j = np.zeros((df.shape[0], nargs, nobs))
        j[n] = df
    return j.T


def jacobs(j):
    """
    unravel jacobian observations
    """
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
