import numpy as np
from DDP.lagrange.equality import augmented_lagrange_equality
from scipy.sparse import csc_matrix
from functools import partial
from time import time
import osqp
from scipy.stats import ortho_group

N = 100 #dimension of variable
M = 50  #contraints number

# def random_SPD_matrix(n):
#     A = np.random.rand(n, n)
#     SPD = A @ A.T
#     return SPD + n * np.eye(n)

def random_SPD_matrix(n):
    return n*2 * (np.random.rand(n) + 1) * np.eye(n)

def LA_QP_solver(P=None, q=None, A=None, b = None):
    """
    A : m * n array
    b : m array
    """
    def f(x):
        return 1/2 * x @ P @ x + q @ x

    def Diffs_f(x):
        return P @ x + q, P

    def c(i, x):
        return A[i] @ x - b[i]

    H_zeros = np.zeros((N, N))

    def Diffs_c(i, x):
        return A[i], H_zeros

    C =       [partial(c, i) for i in range(M)]
    Diffs_C = [partial(Diffs_c, i) for i in range(M)]

    x_0 = np.zeros(N)
    lamb_0 = np.ones(M)
    return augmented_lagrange_equality(f, C, Diffs_f, Diffs_C, x_0, lamb_0)

def comparaison(K: "iterations"):
    T_LA = []
    T_osqp = []

    for _ in range(K):
        P = random_SPD_matrix(N)
        q = N * np.random.rand(N)

        A = M * np.random.rand(M, N)
        b = M * np.random.rand(M)

        t1 = time()
        optimum = LA_QP_solver(P, q, A, b)
        t2 = time()
        t_LA = t2 - t1
        T_LA.append(t_LA)

        print("by LA, x* =", optimum, "process time {}".format(t_LA))

        P = csc_matrix(P)
        A = csc_matrix(A)
        l = b; u = b

        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=l, u=u)
        results = m.solve()

        T_osqp.append(results.info.run_time)
        print("by OSQP, x* =", results.x, "run time {}".format(results.info.run_time))

    avg_LA = sum(T_LA)/K
    avg_osqp = sum(T_osqp)/K

    print("AVERAGE TIME by {} iterations:\n LA: {}, OSQP: {}".format(K, avg_LA, avg_osqp))

comparaison(100)