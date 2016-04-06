import numpy as np
from algopy import UTPM, zeros as zeros, sin, cos
import numdifftools.nd_algopy as nda
import numdifftools as nd

def eval_g(x):
    print x
    print type(x)
    out = zeros(3,dtype=x)
    o0 = 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
    print o0
    print type(o0)
    out[0] = o0
    out[1] = 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
    out[2] = 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]  -4*x[0]**2 - x[1]**2 + 3*x[0]*x[1] -2*x[2]**2 - 5*x[5]+11*x[6]
    return out
    
def eval_jac_g_forward(x):
    x = UTPM.init_jacobian(x)
    return UTPM.extract_jacobian(eval_g(x))

x = np.array([1,2,3,4,0,1,1],dtype=float)

print eval_jac_g_forward(x)
print eval_g(x)

def f(x):
    nobs = x.shape[1:]
    f0 = x[0]**2 * sin(x[1])**2
    f1 = x[0]**2 * cos(x[1])**2
    out = zeros((2,) + nobs, dtype=x)
    out[0,:] = f0
    out[1,:] = f1
    return out

x = np.array([(1, 2, 3, 4),(5, 6, 7, 8)],dtype=float)
y = f(x)

xj = UTPM.init_jacobian(x)
j = UTPM.extract_jacobian(f(xj))

print "x =\n%r\n" % x
print "f =\n%r\n" % y
print "j =\n%r\n" % j

# time it
jaca = nda.Jacobian(f)
x = np.array([np.arange(100),np.random.rand(100)])
%timeit jaca(x)

# x =
# array([[ 1.,  2.,  3.,  4.],
       # [ 5.,  6.,  7.,  8.]])

# f =
# array([[  0.91953576,   0.31229208,   3.88468252,  15.66127584],
       # [  0.08046424,   3.68770792,   5.11531748,   0.33872416]])

# j =
# array([[[ 1.83907153,  0.        ,  0.        ,  0.        , -0.54402111,
          # 0.        ,  0.        ,  0.        ],
        # [ 0.        ,  0.31229208,  0.        ,  0.        ,  0.        ,
         # -2.14629167,  0.        ,  0.        ],
        # [ 0.        ,  0.        ,  2.58978835,  0.        ,  0.        ,
          # 0.        ,  8.9154662 ,  0.        ],
        # [ 0.        ,  0.        ,  0.        ,  7.83063792,  0.        ,
          # 0.        ,  0.        , -4.60645307]],

       # [[ 0.16092847,  0.        ,  0.        ,  0.        ,  0.54402111,
          # 0.        ,  0.        ,  0.        ],
        # [ 0.        ,  3.68770792,  0.        ,  0.        ,  0.        ,
          # 2.14629167,  0.        ,  0.        ],
        # [ 0.        ,  0.        ,  3.41021165,  0.        ,  0.        ,
          # 0.        , -8.9154662 ,  0.        ],
        # [ 0.        ,  0.        ,  0.        ,  0.16936208,  0.        ,
          # 0.        ,  0.        ,  4.60645307]]])

# 100 loops, best of 3: 3.16 ms per loop

# http://numdifftools.readthedocs.org/en/latest/api/generated/numdifftools.core.Jacobian.html

# def g(x):
    # f0 = x[0]**2 + x[1]**2
    # f1 = x[0]**3 + x[1]**3
    # return np.array(zip(f0, f1)).T

# x = np.array([(1.,2.,3.),(4.,5.,6.)])
# jac = nd.Jacobian(g)
# j = jac(x)
