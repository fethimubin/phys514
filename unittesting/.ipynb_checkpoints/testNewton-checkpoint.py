#!/usr/bin/env python

import newton
import unittest
import numpy as N
#import npy_math

class TestNewton(unittest.TestCase): 
    def testLinear(self):
        f = lambda x : 3.0 * x + 6
        solver = newton.Newton(f, tol=1.e-15, maxiter=2)
        x = solver.solve(2)
        self.assertEqual(x, -2.1)

# MY TESTS FOR THE PROVIDED FILES BEFORE ANY ADDITIONS
    def testGeneralFunction(self): 
        # still one dimension, but tests a non-linear function
        f = lambda x : 3.0 / x + 6
        solver = newton.Newton(f, tol=1.e-15, maxiter=5)
        x = solver.solve(-.4)
        self.assertEqual(x, -.5) 

    def testMaxiterTooSmall(self): 
        # tests whether Newton handles unacceptable maxiter:
        # I updated Maxiter such that if maxiter<1
        # maxiter is set to 2
        f = lambda x: newton.F.MyGaussian(x)+10 # is always >1 
        solver = newton.Newton(f, tol=1.e-15, maxiter=-1, r_conv = 1) 
        self.assertEqual(solver._maxiter, 1) 

    def testGeneralFunctionMultiD_1(self): 
        # test multidimensional array with non-linear function 1 
        def f(x): 
            x1 = x[0,0]**3 + x[1,0]**5 +7
            x2 = 1 / x[0,0] + x[1,0]**7 - .5 
            return N.matrix([[x1],[x2]]) 
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(N.matrix("-1.8; .8"))
        N.testing.assert_array_almost_equal(x, N.matrix("-2; 1")) 

    def testGeneralFunctionMultiD_2(self): 
        # test multidimensional array with another non-linear function
        # I use my custom designed function MyNonLinear2D in functions.py
        solver = newton.Newton(newton.F.MyNonLinear2D_1, tol=1.e-15, maxiter=20)
        x = solver.solve(N.matrix(".1; 1.2"))
        N.testing.assert_array_almost_equal(x, N.matrix("0; 1")) 

    def testNonConvergenceError(self): 
        # tests the exception for non-converging Newton iteration.
        solver = newton.Newton(newton.F.MyNonLinear2D_1, tol=1.e-15, maxiter=2)
        
        self.assertRaises(newton.NonConvergenceError, solver.solve, N.matrix(".1;1.2")) 



# TESTS FOR THE ANALYTICAL JACOBIAN
# we utilized the custom written functions in functions.py 
    def testAnalytical_isUsed(self):
        # is analytical Jacobian actually used in Newton?
        A = N.matrix("1. 2.; 3. 4.")
        B = A +1;
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        solver = newton.Newton(newton.F.MyNonLinear2D_1, tol=1.e-15, maxiter=20, dx=1e-6, DfA = B) # feed an analytical J different from the actual one.
        Df_x = solver._DfA 
        N.testing.assert_array_almost_equal(Df_x, B)


    def testAnalytical_doesWork(self):
        # makes sure analytical works just like the numerical in Newton.
        solver = newton.Newton(newton.F.MyNonLinear2D_1, tol=1.e-15, maxiter=20, dx=1e-6, DfA=newton.F.MyNonLinear2D_1_der) 
        x = solver.solve(N.matrix(".1; 1.2"))
        N.testing.assert_array_almost_equal(x, N.matrix("0; 1")) 

#TESTING |x_k - x| < r exception

    def testNonConvergenceErrorInitialAway(self): 
        # tests the exception for non-converging Newton iteration:
        # initial guess is too much away
        solver = newton.Newton(newton.F.MyNonLinear2D_1, tol=1.e-15, maxiter=2, r_conv = .00001) 
        self.assertRaises(newton.DivergingRootError, solver.solve, N.matrix(".1;1.2")) 
        
    def testNonConvergenceErrorFxPositive(self): 
        # tests the exception for non-converging Newton iteration:
        # f(x) > x for all x so x_k increases in each iteration
        f = lambda x: newton.F.MyGaussian(x)+10 # is always >1 
        solver = newton.Newton(f, tol=1.e-15, maxiter=200, r_conv = 1) 
        self.assertRaises(newton.DivergingRootError, solver.solve, 5) 

# EXTRA TESTS

    def testNonConvergenceErrorSingularJacobian(self): 
        # tests the exception for Singular Jacobian:
        # slope is too small
        f = lambda x: newton.F.MyGaussian(x)+10 # is always >1 
        solver = newton.Newton(f, tol=1.e-15, maxiter=200, r_conv = 1) 
        self.assertRaises(newton.SingularJacobianError, solver.solve, 100) 


 
if __name__ == "__main__":
    unittest.main()
