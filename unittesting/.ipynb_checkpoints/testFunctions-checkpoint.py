#!/usr/bin/env python

import functions as F
import numpy as np
import unittest

class TestFunctions(unittest.TestCase):
    def testApproxJacobian1(self):
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x, slope)

    def testApproxJacobian2(self):
        A = np.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = np.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        np.testing.assert_array_almost_equal(Df_x, A) 

    # used less special coordinates to make sure it works, changed
    # asserEqual to assertAlmostEqual to handle double precision
    def testPolynomial(self):
        # p(x) = x^2 + 2x + 3
        p = F.Polynomial([1.3, 2.7, 8.6])
        for x in np.linspace(-2,2,11):
            self.assertAlmostEqual(p(x), 1.3*x**2 + 2.7*x + 8.6)

    # NOVEL TESTS

    def testAnalyticalNumericalCompare_1D(self): 
        # compares analytical and numerical Jacobians for scalars
        x0 = 1.34;
        dx = 1e-10;
        Df_x = F.ApproximateJacobian(F.MyGaussian, x0, dx)
        DfA = F.MyGaussian_der(x0)
        self.assertAlmostEqual(Df_x, DfA) 

    def testAnalyticalNumericalCompare_2D(self): 
        # compares analytical and numerical Jacobians for 2D
        x0 = np.matrix([[1.23],[1.2343]])
        #dx chosen so that dx**2 (rough numerical error) < double precision
        dx = 1e-8;
        Df_x = F.ApproximateJacobian(F.MyNonLinear2D_1, x0, dx)
        DfA = F.MyNonLinear2D_1_der(x0)
        np.testing.assert_array_almost_equal(np.array(Df_x), np.array(DfA)) 


if __name__ == '__main__':
    unittest.main()
