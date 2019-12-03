# a module with some custom written functions.

import numpy as np
import rootfind

def gaussian_custom(x):  # gaussian
    return np.exp(-x**2 /2)

def gaussian_custom_der(x): # derivative of the gaussian
    return -x * np.exp(-x**2 /2) 
 
# A custom implementation of the Lambert W function W(x*exp(x))=x
# method is a string that specifies the root finding algorithm that is used to
# calculate the function, the default is Newton-Raphson iteration.
def lambertw_custom(x, method='newton', tol=1E-15):

    if np.isscalar(x):
        x = np.array([x*1.0])
    
    def x_exp_shifted(z, mask=slice(None)):
        return z * np.exp(z) - x[mask]
    def x_exp_der(z):
        return (z + 1) * np.exp(z)
    #init_guess =1
    init_guess = np.ones_like(x)
    init_guess[x > np.e] = np.log(x[x>np.e])-np.log(np.log(x[x>np.e]))
    #init_guess = np.where(x>np.e,np.log(abs(x))-np.log(abs(np.log(abs(x)))),1)
    #print(init_guess)
    
    #init_guess = init_guess[0]
    if method == 'newton':
        return rootfind.newton(x_exp_shifted, x_exp_der, init_guess, tol)
    elif method == 'newton2':
        return rootfind.newton2(x_exp_shifted, x_exp_der, init_guess, tol)
    elif method == 'halley':
        x_exp_der_der = lambda z: (2*z+1)*np.exp(z)
        return rootfind.halley(x_exp_shifted, x_exp_der, x_exp_der_der, init_guess, tol)
    else:
        printf('Method is not recognized by lambertw_custom !')
        return float('nan')