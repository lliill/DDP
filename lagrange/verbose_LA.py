import numpy as np
import matplotlib.pyplot as plt
# from DDP.Trust_Region.newton_barriers import newton_method
import math
from functools import partial
from time import process_time

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
        Hess_f_x_inv = np.linalg.inv(Hess_f_x)

        decrement = Grad_f_x @ Hess_f_x_inv @ Grad_f_x
        # print("decrement", decrement)
        # print("grad", Grad_f_x)
        if decrement < 0:
            raise Exception("Newton step is not a descent direction!", "full descent:{}".format(-decrement))
        elif decrement/2 <= e:
            # print("decrement", decrement, "e", e)
            return x
        newton_step = -Hess_f_x_inv @ Grad_f_x
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
                                max_iter = 100,
                                verbose = False,
                                plot = False):
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
    x = x_0
    for k in range(max_iter):
        # print(k)
        L_A = partial(augemented_Lagrangian, lamb, nu)
        DL_A = partial(diffs_LA, lamb, nu)

        t1 = process_time()
        x = newton_method(L_A, DL_A, x, e)
        t2 = process_time()
        print("{}th inner problem; nu = {}; process_time = {}")
        
        grad_L = D["gradient_f"] - lamb @ D["constraints_Jacobian"]
        r = D["constraints_residuals"]
        if KKT_test(grad_L, r):
            return x

        lamb = lamb - nu * r
        nu, residual_tolerance = update(nu, r ,residual_tolerance)

if __name__ == '__main__':
    Q = np.array([[2, 0.5],
                  [0.5, 1]])
    p = np.array([1, 1])

    def f(x):
        x1 = x[0]
        x2 = x[1]
        return 2*x1**2 + x2**2 + x1*x2 + x1 + x2

    def Diffs_f(x):
        return 2*Q @ x + p, 2*Q       
    

    def c(x):
        x1 = x[0]
        x2 = x[1]
        return x1 + x2 - 1    

    def Diffs_c(x):
        return np.array([1, 1]), np.zeros((2,2))   

    C = [c]         
    Diffs_C = [Diffs_c]

    x = augmented_lagrange_equality(f, C, Diffs_f, Diffs_C, np.array([0, 0]), np.array([1]))
    print(x)

# test_LA()


