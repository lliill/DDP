import numpy as np
import matplotlib.pyplot as plt
# from DDP.Trust_Region.newton_barriers import newton_method
import math
from functools import partial

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
        # Grad_f_x = gradient_f(x)
        # # hfx = hessian_f(x)
        # Hess_f_x_inv = np.linalg.inv(hessian_f(x))
        Grad_f_x, Hess_f_x = Diffs(x)
        Hess_f_x_inv = np.linalg.inv(Hess_f_x)

        decrement = Grad_f_x @ Hess_f_x_inv @ Grad_f_x
        if decrement/2 <= e:
            return x
        newton_step = -Hess_f_x_inv @ Grad_f_x

        t = line_search(f, x, newton_step, Grad_f_x, a, b)
        x = x + t*newton_step


def augmented_lagrange_equality(f,  
                                C: list,
                                #to add diffs_f, diffs_c
                                # grad_f, hess_f, 
                                # grad_c, hess_c,                                 
                                diffs_f: "function of x, return grad_f_x, hess_f_x", 
                                Diffs_c: "a list of function of x, each one returns grad_ci_x, hess_ci_x", 
                                x_start_0, 
                                lamb_0: "Lagrangian multiplier, type np.array, same dim as constraints",
                                nu_0 : "penalization, type number" = 2, 
                                # tau_0: "residual tolearnce, type number" = 1, 
                                e: "newton tolearnce, type number" = 1e-8,                                                   
                                max_iter = 100,
                                plot = False):
    D = dict()
    # m = len(C)
    def augemented_Lagrangian(lamb: np.array, nu: float, x):
        C_x = np.array([ci(x) for ci in C])
        return f(x) + np.sum((-lamb + nu/2) * C_x) # - sum(lamb * C_x) + nu/2 * sum(C_x) 

    # def grad_x_LA(x, lamb, nu):
    #     return grad_f(x) - sum((lamb[i] - nu * c[i](x)) * grad_c[i](x) for i in range(m))

    # def hess_x_LA(x, lamb, nu):
    #     A = np.array([grad_c[i](x)] for i in range(m)) # jacobian
    #     return hess_f(x) - nu * A.T @ A + sum((lamb[i] + nu *c[i](x)) * hess_c[i](x) for i in range(m))

    def diffs_LA(lamb, nu, x):
        grad_f_x, hess_f_x = diffs_f(x)
        Grad_c_x, Hess_c_x = zip(*(diffs_ci(x) for diffs_ci in Diffs_c))

        # J = np.array([grad_c[i](x)] for i in range(m)) # jacobian_c
        # r = np.array(c[i](x) for i in range(m))
        # grad_x_LA = grad_f_x + (-lamb + nu * r) @ J
        # hess_x_LA = hess_f_x + nu * J.T @ J + sum((-lamb[i] + nu * r[i]) * hess_c[i](x) for i in range(m))
        # return (grad_x_LA, hess_x_LA)
        J = np.array(Grad_c_x) # jacobian_C
        r = np.array([ci(x) for ci in C])
        H = np.array(Hess_c_x) #tensor m * n * n
        grad_x_LA = grad_f_x + (-lamb + nu * r) @ J
        hess_x_LA = hess_f_x + nu * J.T @ J + np.tensordot(-lamb + nu * r, H, axis = 1)

        #* *#
        nonlocal D
        D["constraints_Jacobian"] = J
        D["gradient_f"] = grad_f_x
        D["constraints_residuals"] = r
        #* *#
        return (grad_x_LA, hess_x_LA)

    def norm_inf(x):
        return np.linalg.norm(x, ord = np.inf)

    def KKT_test(grad_L, constraints_residuals):
        norm = np.linalg.norm
        inf = np.inf
        Conditions = []
        Conditions.append( norm(grad_L, ord = inf) < 1e-8 )
        Conditions.append( norm(constraints_residuals, ord = inf) < 1e-8 )
        return all(Conditions)

    def update(penalization, constraints_residuals, residual_tolerance) -> "nu, tau":
        if norm_inf(constraints_residuals) < residual_tolerance:
            return penalization, residual_tolerance/nu**0.9
        else:
            return 100 * penalization, 1/nu**0.1

    lamb = np.array(lamb_0) #important to be np array
    # tau = tau_0
    residual_tolerance = 1
    nu = nu_0
    x_start = x_start_0
    for k in range(max_iter):
        L_A = partial(augemented_Lagrangian, lamb, nu)
        DL_A = partial(diffs_LA, lamb, nu)
        x= newton_method(L_A, DL_A, x_start, nu, e)
        
        grad_L = D["gradient_f"] - lamb @ D["constraints_Jacobian"]
        r = D["constraints_residuals"]
        if KKT_test(grad_L, r):
            return x

        lamb = lamb - nu * r
        nu, residual_tolerance = update(nu, r ,residual_tolerance)




