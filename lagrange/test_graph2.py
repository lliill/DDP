import scipy as sp
import numpy as np 

from scipy.sparse import csc_matrix
from scipy.stats import ortho_group

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


