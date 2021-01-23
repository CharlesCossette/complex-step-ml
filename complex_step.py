import numpy as np
from time import time

def complexStep(f,x,h):
    """ 
    Computes the Jacobian of f with respect to x using the complex step.

    Args:
        f: function f:R^n --> R^m object to differentiate.
        x: derivative evaluation point. Numpy array.
        h: complex step size. float

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    if not type(x) is np.ndarray:
        x = np.array([x])
    n = x.shape[0]
    f_bar = f(x)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e = np.zeros((n,1),dtype=complex)
        e[lv1] = h*1j
        jac[:,lv1] = np.imag(f(x+e))/h
    
    return jac

def centralDifference(f,x,h):
    """ 
    Computes the Jacobian of f with respect to x using central difference.

    Args:
        f: function f:R^n --> R^m object to differentiate.
        x: derivative evaluation point. Numpy array.
        h: central difference step size. float

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    if not type(x) is np.ndarray:
        x = np.array([x])
    n = x.shape[0]
    f_bar = f(x)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e_up = np.zeros((n,1))
        e_down = np.zeros((n,1))
        e_up[lv1] = h
        e_down[lv1] = -h
        jac[:,lv1] = (f(x+e_up)-f(x+e_down))/(2*h)
    
    return jac

def forwardDifference(f,x,h):
    """ 
    Computes the Jacobian of f with respect to x using forward difference.

    Args:
        f: function f:R^n --> R^m object to differentiate.
        x: derivative evaluation point. Numpy array.
        h: forwarddifference step size. float

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    if not type(x) is np.ndarray:
        x = np.array([x])
    n = x.shape[0]
    f_bar = f(x)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e_up = np.zeros((n,1))
        e_up[lv1] = h
        jac[:,lv1] = (f(x+e_up)-f_bar)/(h)
    
    return jac

def testFunction(x):
    return np.sqrt(x.T @ x)

# Derivative evaluation point
x_bar = np.array([1,2,3])
x_bar.shape = (3,1)

# Analytical Jacobian
jac_true = x_bar.T / np.sqrt(x_bar.T @ x_bar)

# Complex step Jacobian 
start = time()
jac_cs = complexStep(testFunction,x_bar,10**-10)
time_cs = time() - start

# Central difference Jacobian
start = time()
jac_cd = centralDifference(testFunction,x_bar,10**-10)
time_cd = time() - start

# Forward difference Jacobian
start = time()
jac_fd = forwardDifference(testFunction,x_bar,10**-6)
time_fd = time() - start

print("Analytical Jacobian:         " + str(jac_true) )
print("Complex-step Jacobian:       " + str(jac_cs)+ ". Elapsed time: " + str(time_cs))
print("Central-difference Jacobian: " + str(jac_cd)+ ". Elapsed time: " + str(time_cd))
print("Forward-difference Jacobian: " + str(jac_fd)+ ". Elapsed time: " + str(time_fd))