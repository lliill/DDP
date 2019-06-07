from cartpole2 import F
import numpy as np
from scipy.optimize import least_squares

N = 100
U_start = np.zeros(N)
X_0 = np.array([0,0,0,0])
WEIGHT = 10000
THETA_desired = np.pi
W_desired = 0

Results = dict()

def common_results(U): #U type tuple or np array?
    



def residuals_J(U):
    """
    U : array_like with shape (n,) 
    """
    N = len(U)
    U = U[:, np.newaxis]
    # X_0 = np.array([0,0,0,0])
    X = [X_0]
    for i in range(1, N+1):
        X_i = F(X[i-1], U[i-1])
        X.append(X_i)
    X_N = X[-1]
    theta_N = X_N[-2]
    w_N = X_N[-1]
    return WEIGHT * np.array([(theta_N - THETA_desired), w_N])

result = least_squares(residuals_J, U_start)