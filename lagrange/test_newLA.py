from scipy import linalg
import numpy as np 

from scipy.sparse import csc_matrix
from scipy.stats import ortho_group
from numpy.linalg import matrix_rank

from newLA import optimize

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

    sol = linalg.solve(KKT, right)
    primal_star, dual_star = sol[:n], sol[n:]
    
    return primal_star, dual_star

def LA_QP_solver(P, q, A, b, primal_sol = None, dual_sol = None):

    M, N = A.shape
    def f(x):
        return 1/2 * x @ P @ x + q @ x

    def gradien_f(x):
        return P @ x + q

    def hessian_f(x): return P

    def c(x):
        return A @ x - b

    H_zeros = np.zeros((M, N, N))

    def jacobian_c(x): return A
        
    def hessian_c(x): return H_zeros

    x_0 = np.zeros(N)

    return optimize(f, gradien_f, hessian_f, c, jacobian_c, hessian_c, x_0, M, #max_penalization = 100,
                            stopping_criteria= {"grad_L" :1e-6,
                                                'constraints_residuals': 1e-6}, 
                                    penalization_increase_rate = 2,
                                    initial_residuals_tolerance = 1e-6,
                                        primal_sol = primal_sol, dual_sol = dual_sol)

P = exp_uniform_SPD_matrix(N)
q = exp_uniform_b(N)
A = exp_uniform_A(M, N)
b = exp_uniform_b(M)

print("A rank:", matrix_rank(A))

primal_star, dual_star = KKT_solve(P, q, A, b)
#primal_osqp, dual_osqp = OSQP_solve(P, q, A, b)
LA_QP_solver(P, q, A, b, primal_sol = primal_star, dual_sol = dual_star)                

