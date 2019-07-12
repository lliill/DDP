import numpy as np
import matplotlib.pyplot as plt
# from DDP.Trust_Region.newton_barriers import newton_method
import math
from functools import partial
from time import time
from scipy import linalg
import matplotlib.pyplot as plt

MAX_PENALIZATION = 1e12

def newton_method(
        f: "convex function to be minimized",
        Diffs: "function of x, return Grad_f_x, Hess_f_x", 
        x: "starting point in the feasible domain of f",
        e: "tolerance, >0" = 1e-8,
        # gradient_f,
        # hessian_f,
        a: "line search parameter, affinity of approximation" = 0.25,
        b: "line search parameter, decreasing rate" = 0.5) -> "x*":
    """Newton's method in the page 487"""
    def line_search(f,
                    x: "starting point in the feasible domain of f",
                    Delta_x: "descent direction",
                    gradient_f_x: "gradient of f at x",
                    a: "affinity of approximation" = 0.25,
                    b: "decreasing rate" = 0.5) -> "step size":
        """Backtracking line search in the Convex Optimization of S.Boyd page 464."""
        t = 1
        while f(x + t*Delta_x) > f(x) + a*t*gradient_f_x @ Delta_x:
            t = b*t
        return t

    while True:
        Grad_f_x, Hess_f_x = Diffs(x)
        # Hess_f_x_inv = np.linalg.inv(Hess_f_x)
        Hg = linalg.solve(Hess_f_x, Grad_f_x)
        newton_step = -Hg

        decrement = Grad_f_x @ Hg
        # print("decrement", decrement)
        # print("grad", Grad_f_x)
        if decrement < 0:
            raise Exception("Newton step is not a descent direction!", "full descent:{}".format(-decrement))
        elif decrement/2 <= e:
            # print("decrement", decrement, "e", e)
            return x
        # newton_step = -Hess_f_x_inv @ Grad_f_x

        # descent = Grad_f_x @ newton_step
        # if descent >= 0:
        #     raise Exception("Newton step is not a descent direction!", "full descent:{}".format(descent))

        t = line_search(f, x, newton_step, Grad_f_x, a, b)
        x = x + t*newton_step


def augmented_lagrange_equality(f,  
                                C: list,                                 
                                diffs_f: "function of x, return grad_f_x, hess_f_x", 
                                Diffs_c: "a list of function of x, each one returns grad_ci_x, hess_ci_x", 
                                x_0, 
                                lamb_0: "Lagrangian multiplier, type np.array, same dim as constraints, >= 1",
                                nu_0 : "penalization, type number" = 2, 
                                # tau_0: "residual tolearnce, type number" = 1, 
                                e: "newton tolearnce, type number" = 1e-8,  
                                stopping_tolerance: "for ||grad L|| and ||c(x)||" = {"grad_L" :1e-6,
                                                            'constraints_residuals': 1e-6},                                                 
                                max_iter = 100,
                                x_star = None,
                                lamb_star = None
                                ) -> "x*":
    D = dict()
    # m = len(C)
    def augemented_Lagrangian(lamb: np.array, nu: float, x):
        C_x = np.array([ci(x) for ci in C])
        return f(x) - lamb @ C_x + np.sum(nu/2 * C_x**2) # - sum(lamb * C_x) + nu/2 * sum(C_x) 

    def diffs_LA(lamb, nu, x):
        grad_f_x, hess_f_x = diffs_f(x)
        Grad_c_x, Hess_c_x = zip(*(diffs_ci(x) for diffs_ci in Diffs_c))

        J = np.array(Grad_c_x) # jacobian_C
        r = np.array([ci(x) for ci in C])
        H = np.array(Hess_c_x) #tensor m * n * n
        grad_x_LA = grad_f_x + (-lamb + nu * r) @ J
        hess_x_LA = hess_f_x + nu * J.T @ J + np.tensordot(-lamb + nu * r, H, axes = 1)

        #* *#
        nonlocal D
        D["constraints_Jacobian"] = J
        D["gradient_f"] = grad_f_x
        D["constraints_residuals"] = r
        #* *#
        return (grad_x_LA, hess_x_LA)

    def norm_inf(x):
        return np.linalg.norm(x, ord = np.inf)

    def KKT_test(grad_L_norm, constraints_residuals_norm):
        # norm = np.linalg.norm
        # inf = np.inf
        Conditions = []
        Conditions.append( grad_L_norm < stopping_tolerance['grad_L'] )
        Conditions.append( constraints_residuals_norm < stopping_tolerance['constraints_residuals'])
        return all(Conditions)

    def update(penalization, constraints_residuals, residual_tolerance) -> "nu, tau":
        if penalization > MAX_PENALIZATION:
            return penalization, residual_tolerance
        if norm_inf(constraints_residuals) < residual_tolerance:
            return penalization, residual_tolerance/penalization**0.9
        else:
            return 10 * penalization, 1/penalization**0.1

    lamb = np.array(lamb_0) #important to be np array
    # tau = tau_0
    residual_tolerance = 1
    nu = nu_0
    x = x_0

    Distances_to_optimum =[]
    Residuals_Norm = []
    Grad_L_Norm = []
    Dual_Distances_to_optimum = []
    Nu = []
    Grad_f_norm = []

    for k in range(max_iter):
        # print(k)
        L_A = partial(augemented_Lagrangian, lamb, nu)
        DL_A = partial(diffs_LA, lamb, nu)
        x = newton_method(L_A, DL_A, x, e)
        
        grad_L = D["gradient_f"] - lamb @ D["constraints_Jacobian"]
        r = D["constraints_residuals"]

        grad_L_norm = norm_inf(grad_L)
        residual_norm = norm_inf(r)

        Distances_to_optimum.append(linalg.norm(x - x_star))
        Dual_Distances_to_optimum.append(norm_inf(lamb - lamb_star))
        Residuals_Norm.append(residual_norm)
        Grad_L_Norm.append(grad_L_norm)
        Nu.append(nu)
        Grad_f_norm.append(norm_inf(D["gradient_f"]))

        if residual_norm <= residual_tolerance:
            if KKT_test(grad_L_norm, residual_norm):
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236)

                ax1.title.set_text("||x - x*||")
                ax2.title.set_text("||c(x)||_inf")
                ax3.title.set_text("||gradient Lagrangian||_inf")
                ax4.title.set_text("||lambda - lambda*||_inf")
                ax5.title.set_text("nu")
                ax6.title.set_text("||gradient_f||_inf")

                ax1.plot(Distances_to_optimum)
                ax2.plot(Residuals_Norm)
                ax3.plot(Grad_L_Norm)
                ax4.plot(Dual_Distances_to_optimum)
                ax5.plot(Nu)
                ax6.plot(Grad_f_norm)

                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax3.set_yscale('log')
                # ax4.set_yscale('log')
                # ax5.set_yscale('log')
                # ax6.set_yscale('log')
                plt.show()
                return x

        lamb = lamb - nu * r
        nu, residual_tolerance = update(nu, r ,residual_tolerance)

# if __name__ == '__main__':
import osqp
from scipy.sparse import csc_matrix

N = 100 #dimension of variable
M = 50  #contraints number

def random_SPD_matrix(n):
    return n*2 * (np.random.rand(n) + 1) * np.eye(n)

def LA_QP_solver(P=None, q=None, A=None, b = None, x_star = None, lamb_star = None):
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
    return augmented_lagrange_equality(f, C, Diffs_f, Diffs_C, x_0, lamb_0, 
                                        x_star = x_star, lamb_star = lamb_star)

def comparaison(K: "iterations"):
    T_LA = []
    T_osqp = []

    for _ in range(K):
        P = random_SPD_matrix(N)
        q = N * np.random.rand(N)

        A = M * np.random.rand(M, N)
        b = M * np.random.rand(M)


        P_csc = csc_matrix(P)
        A_csc = csc_matrix(A)
        l = b; u = b

        m = osqp.OSQP()
        m.setup(P=P_csc, q=q, A=A_csc, l=l, u=u)
        results = m.solve()

        T_osqp.append(results.info.run_time)
        # print("by OSQP, x* =", results.x, "run time {}".format(results.info.run_time))

        x_star = results.x
        lamb_star = results.y


        t1 = time()
        optimum = LA_QP_solver(P, q, A, b, x_star = x_star, lamb_star = lamb_star)
        t2 = time()
        t_LA = t2 - t1
        T_LA.append(t_LA)

        print("by LA, x* =", optimum, " time {}".format(t_LA))

    avg_LA = sum(T_LA)/K
    avg_osqp = sum(T_osqp)/K

    print("AVERAGE TIME by {} iterations:\n LA: {}, OSQP: {}".format(K, avg_LA, avg_osqp))

comparaison(1)