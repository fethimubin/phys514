# newton - Newton-Raphson solver
#
# For APC 524 Homework 3
# CWR, 18 Oct 2010

import numpy as N
import functions as F

""" Errors raised by Newton"""
class NonConvergenceError(Exception): pass 

class DivergingRootError(Exception): pass 

class SingularJacobianError(Exception): pass 


class Newton(object):
    def __init__(self, f, tol=1.e-6, maxiter=20, dx=1.e-6, DfA = None, r_conv =None):
        """Return a new object to find roots of f(x) = 0 using Newton's method.
        tol:     tolerance for iteration (iterate until |f(x)| < tol)
        maxiter: maximum number of iterations to perform
        dx:      step size for computing approximate Jacobian"""
        self._f = f
        self._tol = tol
        if maxiter < 1: 
            print """ maximum number of iterations cannot be less than 1.
Updating number of maximum iterations to 1"""
            maxiter = 1 
        
        self._maxiter = maxiter
        self._dx = dx 
        self._DfA = DfA #optional analytical Jacobian
        self._r_conv = r_conv;

    def solve(self, x0):
        """Return a root of f(x) = 0, using Newton's method, starting from
        initial guess x0"""
        x = x0
                
        for i in xrange(self._maxiter+1): #add 1 
            fx = self._f(x)
            if N.linalg.norm(fx) < self._tol:
                return x
            x = self.step(x, fx)
            if (self._r_conv != None) and (N.linalg.norm(x-x0) > self._r_conv): 
                print "ITERATED ROOT TOO MUCH AWAY FROM BEGINNING"
                raise DivergingRootError
            
        """ raise exception if final f(x) is not within tolerance"""
        if N.linalg.norm(self._f(x)) < self._tol:
            return x
        else: 
            print """WARNING: NEWTON RAPHSON SCHEME DID NOT CONVERGE in %s ITERATIONS""" %(self._maxiter) 
            raise NonConvergenceError
    
    def step(self, x, fx=None):
        """Take a single step of a Newton method, starting from x
        If the argument fx is provided, assumes fx = f(x)"""
        if fx is None:
            fx = self._f(x)
        
        # if analytical Jacobian is not provided, calculate the 
        # Jacobian numerically; otherwise use the analytic one.
        if self._DfA is None: 
            Df_x = F.ApproximateJacobian(self._f, x, self._dx) 
        else:   
            Df_x = self._DfA(x)  
        
        try: 
            h = N.linalg.solve(N.matrix(Df_x), N.matrix(fx)) 
        except N.linalg.LinAlgError : 
            print "JACOBIAN CANNOT BE INVERTED"
            raise SingularJacobianError

        return x - h # BUG FIXED: + changed to -
