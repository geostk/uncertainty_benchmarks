
import numpy as np
import numdifftools as nd
import numdifftools.nd_algopy as nda
from uncertainties import Variable, ufloat, umath, unumpy, correlated_values
from statsmodels.tools import numdiff
from scipy.misc import derivative
import quantities as pq
from time import clock
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

AVG = np.random.rand(1000) * 11.
COV = np.random.rand(1000, 1000) / 11.
COV *= COV.T
TOL = 1e-6
COV = np.where(COV > TOL, COV, np.zeros((1000, 1000)))
STD = np.sqrt(COV.diagonal())

# test pure numpy implementation of uncertainty


def test_puro_cov(avg=AVG, cov=COV):
    c = clock()
    nobs = avg.size
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = np.eye(nobs) * np.cos(avg.flatten())
    LOGGER.debug('\t2> %g [s]', clock() - c)
    cov = np.dot(np.dot(j, cov), j.T)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = np.sin(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    cov = np.reshape(cov.diagonal(), avg.shape)
    LOGGER.debug('\t5> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('cov', float)])
    LOGGER.debug('\t6> %g [s]', clock() - c)
    return np.array(zip(avg, cov), dtype=dt)
    

cstart = clock()
np.sin(AVG)
cstop = clock()
LOGGER.debug('calculate sin(avg):\n\telapsed time> %g [s]\n', cstop - cstart)

cstart = clock()
np.cos(AVG)
cstop = clock()
LOGGER.debug('calculate cos(avg):\nelapsed time> %g [s]\n', cstop - cstart)

cstart = clock()
r1 = test_puro_cov()
cstop = clock()
LOGGER.debug('test puro w/covariance:\n\telapsed time> %g [s]\n',
             cstop - cstart)


def test_puro_std(avg=AVG, std=STD):
    c = clock()
    j = np.cos(avg)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    std = np.abs(j*std)  # np.sqrt(j*std*std*j)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    avg = np.sin(avg)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('std', float)])
    LOGGER.debug('\t4> %g [s]', clock() - c)
    return np.array(zip(avg, std), dtype=dt)


cstart = clock()
r2 = test_puro_std()
cstop = clock()
LOGGER.debug('test puro w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

# test uncertainties package

cstart = clock()
X = unumpy.uarray(AVG, STD)
cstop = clock()
LOGGER.debug('create ufloat array w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)


def test_uncertainties_std(x=X):
    return unumpy.sin(x)


cstart = clock()
r3 = test_uncertainties_std()
cstop = clock()
LOGGER.debug('test uncertainties w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

cstart = clock()
Y = correlated_values(AVG, COV)
cstop = clock()
LOGGER.debug('create array of correlated values:\n\telapsed time> %g [s]\n',
             cstop - cstart)


def test_uncertainties_cov(y=Y):
    return unumpy.sin(y)


cstart = clock()
r4 = test_uncertainties_cov()
cstop = clock()
LOGGER.debug('test uncertainties w/ covariance:\n\telapsed time> %g [s]\n',
             cstop - cstart)

F = lambda x: np.sin(x)
EPS = np.finfo(float).eps
DX = EPS ** (1. / 3.)


# test numerical differentiation methods

# scipy.misc.derivative
# * inputs to must be an array

def test_scipy_derivative(avg=AVG, std=STD, f=F):
    c = clock()
    j = derivative(f, avg, dx=DX)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    std = np.abs(j*std)  # np.sqrt(j*std*std*j)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    avg = F(avg)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('std', float)])
    LOGGER.debug('\t4> %g [s]', clock() - c)
    return np.array(zip(avg, std), dtype=dt)


def test_statsmodels_numdiff_std(avg=AVG, std=STD, f=F):
    c = clock()
    j = numdiff.approx_fprime(avg, f, centered=True)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    std = np.abs(j.diagonal()*std)  # np.sqrt(j*std*std*j)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('std', float)])
    LOGGER.debug('\t4> %g [s]', clock() - c)
    return np.array(zip(avg, std), dtype=dt)


def test_statsmodels_numdiff_cov(avg=AVG, cov=COV, f=F):
    c = clock()
    j = numdiff.approx_fprime(avg, f, centered=True)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    cov = np.dot(np.dot(j, cov), j.T)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    cov = np.reshape(cov.diagonal(), avg.shape)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('cov', float)])
    LOGGER.debug('\t5> %g [s]', clock() - c)
    return np.array(zip(avg, cov), dtype=dt)


cstart = clock()
r5 = test_scipy_derivative()
cstop = clock()
LOGGER.debug('test scipy derivative w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

cstart = clock()
r6 = test_statsmodels_numdiff_std()
cstop = clock()
LOGGER.debug('test statsmodels numdiff w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

cstart = clock()
r7 = test_statsmodels_numdiff_cov()
cstop = clock()
LOGGER.debug('test statsmodels numdiff w/covariance:\n\telapsed time> %g [s]\n',
             cstop - cstart)

# compare averages
print "compare avg r1 to r2: %s" % np.allclose(r1['avg'], r2['avg'])
print "compare avg r1 to r3: %s" % np.allclose(r1['avg'], [_.n for _ in r3])
print "compare avg r1 to r4: %s" % np.allclose(r1['avg'], [_.n for _ in r4])
print "compare avg r1 to r5: %s" % np.allclose(r1['avg'], r5['avg'])
print "compare avg r1 to r6: %s" % np.allclose(r1['avg'], r6['avg'])
print "compare avg r1 to r7: %s" % np.allclose(r1['avg'], r7['avg'])
print

# compare std-dev
print "compare std r2 to r3: %s" % np.allclose(r2['std'], [_.s for _ in r3])
print "compare std r2 to r5: %s" % np.allclose(r2['std'], r5['std'])
print "compare std r2 to r6: %s" % np.allclose(r2['std'], r6['std'])
print

# compare covariance
print "compare cov r1 to r4: %s" % np.allclose(r1['cov'], np.array([_.s for _ in r4]) ** 2)
# XXX: *** Uncertainties covariance doesn't match expected values ***
print "compare cov r1 to r7: %s" % np.allclose(r1['cov'], r7['cov'])
print
