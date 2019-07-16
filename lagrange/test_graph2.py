import scipy as sp
import numpy as np 

from scipy.sparse import csc_matrix
from scipy.stats import ortho_group
from numpy.linalg import matrix_rank

from test_graph import augmented_lagrange_equality, LA_QP_solver

N = 100 #dimension of variable
M = 50  #contraints number

def exp_uniform_SPD_matrix(n, Max = 1000, Min = 0.1):
    """
    une distribution sur (Min, Max). i.e. la loi est la mesure image de mesure uniforme par exponentielle sur (Min, Max)
    """
    x = ortho_group.rvs(n)
    c = np.diag(np.exp((np.log(Max) - np.log(Min)) * np.random.rand(n) + np.log(Min)))
    return x @ c @ x.T 

def exp_uniform_A(p, n, Max = 1000, Min = 0.1):
    return np.exp((np.log(Max) - np.log(Min)) * np.random.rand(p, n) + np.log(Min))

def exp_uniform_b(p, Max = 1000, Min = 0.1):
    return np.exp((np.log(Max) - np.log(Min)) * np.random.rand(p) + np.log(Min))

def KKT_solve(P, q, A, b):

    p, n = A.shape
    up = np.concatenate((P, A.T), axis= 1) 
    bottom = np.pad(A, ((0,0),(0,p)),'constant')  
    KKT  = np.concatenate((up, bottom), axis= 0) 
    right = np.concatenate((-q, b))

    sol = sp.linalg.solve(KKT, right)
    primal_star, dual_star = sol[:n], sol[n:]
    
    return primal_star, dual_star

def OSQP_solve(P, q, A, b):
    import osqp
    from scipy.sparse import csc_matrix

    P_csc = csc_matrix(P)
    A_csc = csc_matrix(A)
    l = b; u = b

    m = osqp.OSQP()
    m.setup(P=P_csc, q=q, A=A_csc, l=l, u=u)
    results = m.solve()

    return results.x, results.y


P = exp_uniform_SPD_matrix(100)
q = exp_uniform_b(100)
A = exp_uniform_A(50, 100)
b = exp_uniform_b(50)

print("A rank:", matrix_rank(A))

primal_star, dual_star = KKT_solve(P, q, A, b)
#primal_osqp, dual_osqp = OSQP_solve(P, q, A, b)
LA_QP_solver(P, q, A, b, x_star = primal_star, lamb_star = dual_star)



