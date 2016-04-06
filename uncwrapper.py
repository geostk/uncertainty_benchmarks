import numpy as np
#from algopy import UTPM, zeros as azeros, exp as aexp
#import numdifftools.nd_algopy as nda
#import numdifftools as nd
from uncertainties import ufloat, umath, unumpy, Variables
# TODO: map unumpy ufuncs to Variables
from statsmodels.tools import numdiff
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# f = lambda t: np.concatenate([np.reshape(t[0] * np.exp(t[1]),(1,-1)), np.reshape(t[1] * np.exp(t[0]),(1,-1))])


def f(t):
    # t must be a sequence
    nobs = t.shape[1:]
    logger.debug('nobs: %r', nobs)
    f0 = t[0] * np.exp(t[1])
    logger.debug('f0\n\t%r', f0)
    f1 = t[1] * np.exp(t[0])
    logger.debug('f1\n\t%r', f1)
    out = azeros((2, 2), dtype=x)
    out[0,:] = f0
    out[1,:] = f1
    return out


def unc(cov):
    def unc_wrapper(f):
        jac = nda.Jacobian(f)
        def wrapper(x, *args, **kwargs):
            x = np.asarray(x)
            avg = f(x, *args, **kwargs)
            logger.debug('avg:\n\t%r', avg)
            if avg.ndim == 0:
                avg = np.reshape(avg, (1,))
            j = jac(x, *args, **kwargs)
            logger.debug('jac:\n\t%r', j)
            var = np.dot(np.dot(j, cov), j.T)
            logger.debug('var:\n\t%r', var)
            nobs = x.shape[1:]  # number of observations for each independent
            dt = np.dtype([('avg', 'float', nobs), ('var', 'float', nobs)])
            return np.array(zip(avg, var), dtype=dt)
        return wrapper
    return unc_wrapper

g = unc(np.array([0.1, 0.2]))(f)
g(np.array([1.0, 2.0]))

#x = np.array([(1,2,3),(4,5,6)])