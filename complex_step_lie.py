import numpy as np
import math

def expmTaylor(A):
    """
    Computes the matrix exponential of a matrix A using a 
    direct Taylor series expansion until convergence. 
    
    Args:
        A: n x n matrix. ndarray
    Returns:
        X: n x n matrix exponential of A. ndarray
    """

    n = A.shape[0]
    X_old = 1000*np.eye(n)
    X = np.zeros((n,n))
    k = 0
    while np.linalg.norm(X - X_old,'fro') > 1e-15:
        X_old = X
        X = X + np.linalg.matrix_power(A,k)/math.factorial(k)
        k = k + 1
    return X

def complexStepLie(f,X,n,wedgeOp):
    """ 
    Computes the Jacobian of f with respect to X using the complex step.

    Args:
        f: function f:G --> R^m object to differentiate.
        X: derivative evaluation point. Numpy array.
        n: Dimension (degree of freedom) of G. Int.
        wedgeOp: function wedgeOp: R^n --> g from R^n to the Lie algebra

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    h = 1e-18
    f_bar = f(X)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e = np.zeros((n,1),dtype=complex)
        e[lv1] = h*1j
        jac[:,lv1] = np.imag(f(expmTaylor(wedgeOp(e)) @ X))/h
    
    return jac

def centralDifferenceLie(f,X,n,wedgeOp):
    """ 
    Computes the Jacobian of f with respect to Xusing central difference.

    Args:
        f: function f:G --> R^m object to differentiate.
        X: derivative evaluation point. Numpy array.
        n: Dimension (degree of freedom) of G. Int.
        wedgeOp: function wedgeOp: R^n --> g from R^n to the Lie algebra

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    h = 1e-8
    f_bar = f(X)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e_up = np.zeros((n,1))
        e_down = np.zeros((n,1))
        e_up[lv1] = h
        e_down[lv1] = -h
        jac[:,lv1] = (f(expmTaylor(wedgeOp(e_up)) @ X)-f(expmTaylor(wedgeOp(e_down)) @ X))/(2*h)
    
    return jac

def forwardDifferenceLie(f,X,n,wedgeOp):
    """ 
    Computes the Jacobian of f with respect to X using forward difference.

    Args:
        f: function f:G --> R^m object to differentiate.
        X: derivative evaluation point. Numpy array.
        n: Dimension (degree of freedom) of G. Int.
        wedgeOp: function wedgeOp: R^n --> g from R^n to the Lie algebra

    Returns:
        jac: m x n ndarray Jacobian of f with respect to x
    """
    h = 1e-6
    f_bar = f(X)
    m = f_bar.shape[0]

    jac = np.zeros((m,n))

    for lv1 in range(n):
        e_up = np.zeros((n,1))
        e_up[lv1] = h
        jac[:,lv1] = (f(expmTaylor(wedgeOp(e_up)) @ X)-f_bar)/(h)
    
    return jac

def testFunction(X):
    # Function to differentiate
    v = np.array([3,4,5])
    y = np.array([1,2,3])
    y.shape = (3,1)
    return v @ X @ y

def wedgeSO3(x):
    """
    Skew-symmetrix cross operator for SO(3). Takes element of R^n
    to the Lie algebra.

    Args:
        x: 3 x 1 ndarray
    """
    return np.array([[0, -x[2, 0], x[1, 0]],[x[2, 0],0,-x[0, 0]],[-x[1, 0],x[0, 0],0]])


##### TEST
phi_hat = np.array([0,0.5,0.2])
phi_hat.shape = (3,1)
Xi = wedgeSO3(phi_hat)
C_bar = expmTaylor(Xi)

# Analytical Jacobian
v = np.array([3,4,5])
y = np.array([1,2,3])
y.shape = (3,1)
jac_true = -v @ wedgeSO3(C_bar @ y)

# Complex step Jacobian
jac_cs = complexStepLie(testFunction,C_bar,3,wedgeSO3)
jac_cd = centralDifferenceLie(testFunction,C_bar,3,wedgeSO3)
jac_fd = forwardDifferenceLie(testFunction,C_bar,3,wedgeSO3)

print("Analytical Jacobian:         " + str(jac_true) )
print("Complex-step Jacobian:       " + str(jac_cs))
print("Central-difference Jacobian: " + str(jac_cd))
print("Forward-difference Jacobian: " + str(jac_fd))