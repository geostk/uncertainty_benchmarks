"""
Notes on scipy.misc.derivative and statsmodels.tools.numdiff

https://github.com/scipy/scipy/blob/master/scipy/misc/common.py
* scipy uses central difference
* simplest case for dx = eps^0.3, n = 1 and order = 3 this is the algorithm:

    >>> df = lambda x: f(x + dx/2, *args) - f(x - dx/2, *args) / dx

https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tools/numdiff.py
* statsmodels also uses central difference

"""
import numpy as np
from statsmodels.tools import numdiff
from scipy.misc import derivative

F = lambda x: np.array([np.sin(x), np.cos(x)])
X = np.random.rand(4)

F(X)
# array([[ 0.34751839,  0.1399848 ,  0.16636918,  0.8145621 ],
#        [ 0.93767317,  0.99015365,  0.98606354,  0.58007637]])

EPS = np.finfo(float).eps
DX = EPS ** (1. / 3.)

derivative(F, X, dx=DX)
# array([[ 0.93767317,  0.99015365,  0.98606354,  0.58007637],
#        [-0.34751839, -0.1399848 , -0.16636918, -0.8145621 ]])

numdiff.approx_fprime(X, F, centered=True)
#array([[[ 0.93767317,  0.        ,  0.        ,  0.        ],
#        [-0.34751839,  0.        ,  0.        ,  0.        ]],
#
#       [[ 0.        ,  0.99015365,  0.        ,  0.        ],
#        [ 0.        , -0.1399848 ,  0.        ,  0.        ]],
#
#       [[ 0.        ,  0.        ,  0.98606354,  0.        ],
#        [ 0.        ,  0.        , -0.16636918,  0.        ]],
#
#       [[ 0.        ,  0.        ,  0.        ,  0.58007637],
#        [ 0.        ,  0.        ,  0.        , -0.8145621 ]]])