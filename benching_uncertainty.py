"""
benchmark uncertainty methods with simple function

============================  ==============
method                        Benchmark [s]
============================  ==============
Numpy w/std-dev only          0.000866898
create ufloat w/std-dev only  0.00197498
Uncertainties w/std-dev only  0.00774529 ($)
Scipy w/std-dev only          0.00157711
statsmodels w/std-dev only    0.0441201
numdifftools w/std-dev only   0.104556
algopy w/std-dev              0.0277903 (*)
----------------------------  --------------
Numpy w/covariance            0.0770831
Jacobian estimate             0.0816991
statsmodels w/covariance      0.113367
numdifftools w/covariance     85.104
algopy w/covariance           0.156879
============================  ==============

Notes:
------
* Uncertainties covariance doesn't work with wrapped functions and not with
  unumpy vectorized analytical derivatives either, fails comparison 
* variance of all runtimes is large, from 10% up to 50%
($) total time for Uncertainties is 0.00972072[s]
(*) algopy derivative runtimes seems to vary by 10x, weird!
"""

import numpy as np
import numdifftools as nd
import numdifftools.nd_algopy as nda
from algopy import sin
from uncertainties import ufloat, umath, unumpy, correlated_values, wrap
#
# wrap only works on scalars
# from example in help(wrap)
# >>> f_wrapped = wrap(math.sin)
# >>> f_wrapped(ufloat(1.23, 0.45))
# 0.9424888019316975+/-0.15040697757063842
# try with np.array of Variable objects fails with
#
#     TypeError: only length-1 arrays can be converted to Python scalars
#
from statsmodels.tools import numdiff
from scipy.misc import derivative
from jacobian_fun import jacobian, jacobs
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
# * can handle multiple observations but not multiple args
# * ie: f(x) must be a scalar function
# * however f(x) return can be an array
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

# test Jacobian estimate
def test_jacobian_cov(avg=AVG, cov=COV, f=F):
    c = clock()
    avg = np.atleast_2d(avg)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = jacobs(jacobian(f, avg))
    LOGGER.debug('\t2> %g [s]', clock() - c)
    cov = np.dot(np.dot(j, cov), j.T)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    cov = np.reshape(cov.diagonal(), avg.shape)
    LOGGER.debug('\t5> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('cov', float)])
    LOGGER.debug('\t6> %g [s]', clock() - c)
    return np.array(zip(avg.squeeze(), cov.squeeze()), dtype=dt)


cstart = clock()
r12 = test_jacobian_cov()
cstop = clock()
LOGGER.debug('test jacobian estimate:\n\telapsed time> %g [s]\n',
             cstop - cstart)

# statsmodels.tools.models
# * numdiff seems to have problems broadcasting despite what its docs say
# * x must be a 1-d array apparently
# * if it is 2-d or if indexing produces a sequence it fails
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

# test numdifftools


def test_numdifftools_std(avg=AVG, std=STD, f=F):
    c = clock()
    jac = nd.Derivative(f)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = jac(avg)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    std = np.abs(j*std)  # np.sqrt(j*std*std*j)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('std', float)])
    LOGGER.debug('\t5> %g [s]', clock() - c)
    return np.array(zip(avg, std), dtype=dt)


def test_numdifftools_cov(avg=AVG, cov=COV, f=F):
    c = clock()
    jac = nd.Jacobian(f)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = jac(avg)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    cov = np.dot(np.dot(j, cov), j.T)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    cov = np.reshape(cov.diagonal(), avg.shape)
    LOGGER.debug('\t5> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('cov', float)])
    LOGGER.debug('\t6> %g [s]', clock() - c)
    return np.array(zip(avg, cov), dtype=dt)


cstart = clock()
r8 = test_numdifftools_std()
cstop = clock()
LOGGER.debug('test numdifftools w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

cstart = clock()
r9 = test_numdifftools_cov()
cstop = clock()
LOGGER.debug('test numdifftools w/covariance:\n\telapsed time> %g [s]\n',
             cstop - cstart)

# test numdifftools with algopy

G = lambda x: sin(x)

def test_algopy_std(avg=AVG, std=STD, f=G):
    c = clock()
    jac = nda.Derivative(f)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = jac(avg)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    std = np.abs(j*std)  # np.sqrt(j*std*std*j)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('std', float)])
    LOGGER.debug('\t5> %g [s]', clock() - c)
    return np.array(zip(avg, std), dtype=dt)


def test_algopy_cov(avg=AVG, cov=COV, f=G):
    c = clock()
    jac = nda.Jacobian(f)
    LOGGER.debug('\t1> %g [s]', clock() - c)
    j = jac(avg)
    LOGGER.debug('\t2> %g [s]', clock() - c)
    cov = np.dot(np.dot(j, cov), j.T)
    LOGGER.debug('\t3> %g [s]', clock() - c)
    avg = f(avg)
    LOGGER.debug('\t4> %g [s]', clock() - c)
    cov = np.reshape(cov.diagonal(), avg.shape)
    LOGGER.debug('\t5> %g [s]', clock() - c)
    dt = np.dtype([('avg', float), ('cov', float)])
    LOGGER.debug('\t6> %g [s]', clock() - c)
    return np.array(zip(avg, cov), dtype=dt)


cstart = clock()
r10 = test_algopy_std()
cstop = clock()
LOGGER.debug('test algopy w/std-dev only:\n\telapsed time> %g [s]\n',
             cstop - cstart)

cstart = clock()
r11 = test_algopy_cov()
cstop = clock()
LOGGER.debug('test algopy w/covariance:\n\telapsed time> %g [s]\n',
             cstop - cstart)

# compare averages
print "compare avg numpy to numpy: %s" % np.allclose(r1['avg'], r2['avg'])
print "compare avg numpy to numpy: %s" % np.allclose(
    r1['avg'], [_.n for _ in r3])
print "compare avg numpy to Uncertainties: %s" % np.allclose(
    r1['avg'], [_.n for _ in r4])
print "compare avg numpy to scipy: %s" % np.allclose(r1['avg'], r5['avg'])
print "compare avg numpy to statsmodels: %s" % np.allclose(r1['avg'], r6['avg'])
print "compare avg numpy to statsmodels: %s" % np.allclose(r1['avg'], r7['avg'])
print "compare avg numpy to numdifftools: %s" % np.allclose(r1['avg'], r8['avg'])
print "compare avg numpy to numdifftools: %s" % np.allclose(r1['avg'], r9['avg'])
print "compare avg numpy to algopy: %s" % np.allclose(r1['avg'], r10['avg'])
print "compare avg numpy to algopy: %s" % np.allclose(r1['avg'], r11['avg'])
print

# compare std-dev
print "compare std numpy to Uncertainties: %s" % np.allclose(
    r2['std'], [_.s for _ in r3])
print "compare std numpy to scipy: %s" % np.allclose(r2['std'], r5['std'])
print "compare std numpy to statsmodels: %s" % np.allclose(r2['std'], r6['std'])
print "compare std numpy to numdifftools: %s" % np.allclose(r2['std'], r8['std'])
print "compare std numpy to algopy: %s" % np.allclose(r2['std'], r10['std'])
print

# compare covariance
print "compare cov numpy to Uncertainties: %s" % np.allclose(
    r1['cov'], np.array([_.s for _ in r4]) ** 2)
# XXX: *** Uncertainties covariance doesn't match expected values ***
print "compare cov numpy to statsmodels: %s" % np.allclose(r1['cov'], r7['cov'])
print "compare cov numpy to numdifftools: %s" % np.allclose(r1['cov'], r9['cov'])
print "compare cov numpy to algopy: %s" % np.allclose(r1['cov'], r11['cov'])
print "compare cov numpy to jacobian estimate: %s" % np.allclose(
    r1['cov'], r12['cov'])
print
