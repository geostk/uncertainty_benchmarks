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
    # other ideas:
    # delta = np.eye(nargs) * DELTA
    # delta = np.daig(np.repeat(DELTA, nargs))
    # delta = np.daig(DELTA * np.ones(nargs))
    # for d in delta:
    #    pass
    # delta = np.array([DELTA] + [0] * (nargs - 1))
    delta = np.pad([[DELTA]], ((0, nargs - 1), (0, 0)), 'constant')
    delta = delta.repeat(nobs, axis=1)
    for n in xrange(nargs):
        dx = 2.0 * DELTA * np.tile(x[n], (nargs, 1))
        df = (f(x * (1.0 + delta)) - f(x * (1.0 - delta))) / dx
        # derivatives df/d_n
        if j is None:
            j = df.reshape(1, nargs, nobs)
        else:
            j = np.append(j, df.reshape(1, nargs, nobs), axis=0)
        delta = np.roll(delta, 1, axis=0)
    return j

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
    print jacobian(g, z)
    print "elapsed time = %g [s]" % (clock() - c)
    print "==================================================================="
