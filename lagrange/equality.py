import numpy as np
import matplotlib.pyplot as plt
# from DDP.Trust_Region.newton_barriers import newton_method
import math

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
    def log(x):
        try:
            return math.log(x)
        except ValueError:
            return -math.inf


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
                                c: list,
                                #to add diffs_f, diffs_c
                                grad_f, hess_f,
                                grad_c, hess_c,
                                x_start_0, 
                                nu_0 : "", 
                                tau_0,                    
                                lamb_0,
                                max_iter = 100,
                                plot = False):

    m = len(c)
    def augemented_Lagrangian(x, lamb, nu):
        return f(x) - sum(lamb[i]*c[i](x) for i in range(m)) + nu/2 * sum(c[i](x) for i in range(m))

    # def grad_x_LA(x, lamb, nu):
    #     return grad_f(x) - sum((lamb[i] - nu * c[i](x)) * grad_c[i](x) for i in range(m))

    # def hess_x_LA(x, lamb, nu):
    #     A = np.array([grad_c[i](x)] for i in range(m)) # jacobian
    #     return hess_f(x) - nu * A.T @ A + sum((lamb[i] + nu *c[i](x)) * hess_c[i](x) for i in range(m))

    def diffs_x_LA(x, lamb, nu):
        J = np.array([grad_c[i](x)] for i in range(m)) # jacobian_c
        r = np.array(c[i](x) for i in range(m))
        grad_x_LA = grad_f(x) + (-lamb + nu * r) @ J
        hess_x_LA = hess_f(x) + nu * J.T @ J + sum((-lamb[i] + nu * r[i]) * hess_c[i](x) for i in range(m))
        return (grad_x_LA, hess_x_LA)

    for k in range(max_iter):

        x = 1
