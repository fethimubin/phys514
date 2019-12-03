# a module of nonlinear root finding algortihms
import numpy as np
from warnings import warn

# newton iteration with optional initial guess x
def newton(fun, dfun, x=0, tol=1E-15, max_steps=20):
    for _ in range(20):
        fx = fun(x) # why assign a variable? to avoid calculating fun(x)
                    # more than once, which might be costly
        if np.abs(fx).max() < tol:
            return x
        x -= fx/dfun(x)
    return x

# newton iteration with optional initial guess x
def newton2(fun, dfun, x=0, tol=1E-15, max_steps=20):
    converged = np.zeros_like(x, dtype='bool')
    for _ in range(max_steps):
        fx = fun(x[~converged], ~converged)
        flipped = np.abs(fx) < tol
        converged[~converged] = flipped
        if np.all(converged):
            return x
        x[~converged] -= fx[~flipped] / dfun(x[~converged])
    warn(f"Max steps reached w/o convergence; increase tol or max_steps")
    return x

# newton iteration with optional initial guess x
def newton3(fun, fun_over_dfun, x=0, tol=1E-15, max_steps=20):
    for _ in range(20):
        fx = fun(x) # why assign a variable? to avoid calculating fun(x)
                    # more than once, which might be costly
        if np.abs(fx).max() < tol:
            return x
        x -= fun_over_dfun(x)
    return x

# halley iteration with optional initial guess x
def halley(fun, dfun, ddfun, x=0, tol=1E-15, max_steps=20):
    for _ in range(20):
        fx = fun(x)
        if np.abs(fx).max() < tol:
            return x
        x -= (2*fx*dfun(x))/(2*dfun(x)**2-fx*ddfun(x))
    return x

        