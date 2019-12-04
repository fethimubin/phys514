import numpy as N

# trial for hg

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1 
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):
        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/dx # BUG FIX: divide by dx
    return Df_x


class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c
        return ans

    def __call__(self, x):
        return self.f(x)

# extra nonlinear functions and their analytical derivatives for testing

def MyGaussian(x):  # gaussian
    return N.exp(-x**2 /2)

def MyGaussian_der(x): # derivative of the gaussian
    return -x * N.exp(-x**2 /2) 
 
def MyNonLinear2D_1(x): 
    x1 = N.exp(x[0,0]) + N.sin(N.arcsin(1) * x[1,0])+  x[0,0]*x[1,0] -2
    x2 = 1/(1+x[0,0]**2) + N.log(x[1,0]) +N.exp(x[0,0]+x[1,0]) - 1 - N.exp(1) 
    return N.matrix([[x1],[x2]]) 

def MyNonLinear2D_1_der(x): 
    y11 = N.exp(x[0,0])+  x[1,0] 
    y12 = N.arcsin(1)* N.cos(N.arcsin(1) * x[1,0])+  x[0,0] 
    y21 = -2*x[0,0]/(1+x[0,0]**2)**2 + N.exp(x[0,0]+x[1,0]) 
    y22 = 1 / x[1,0] +N.exp(x[0,0]+x[1,0]) 
    return N.matrix([[y11, y12],[y21 , y22]]) 
