# Trust region by hessian regularization
import numpy as np
import matplotlib.pyplot as plt
from math import isclose
from newton_barriers import newton_method, line_search

def solve_subproblem(f_k: "f(x_k)", 
                    g_k: "Grad f(x_k)",
                    B_k: "B_k symmetic and uniformly bounded, Aprox Hess of f(x_k)",
                    nu_k: "regularization weight") -> "p_k":
    """Solve min_p m(p) = f_k + g_k^T p + 1/2 p^T (B_k + nu_k*Id) p 
                
        the type of g needs to support @ operation, like 1-D np array"""
    
    n = np.size(g_k)
    H_k = B_k + nu_k * np.identity(n)
    def m(p):
        return f_k + g_k @ p + 1/2 * p @ H_k @ p

    def grad_m(p):
        return g_k + H_k @ p

    def hess_m(p):
        return H_k

    p0 = np.arange(n)
    tolerance = 1e-8
    return newton_method(m, p0, tolerance, grad_m, hess_m)


def regularization(fun,
                   x0: np.array,
                   gradient_f,
                   approx_hessian_f,
                   iterations: int,
                   minimal_regularization: float = 1e-6,
                   initial_regularization: float = 1,
                   minimal_modification_factor: float = 2,
                   initial_modification_factor: float = 2,
                   ok_ratio: float = 0.2) -> "x*":
    """Compute the local minimum by Regularization method. 

    Arguments:
    fun -- the function to be optimized
    x0 -- the initial feasible starting point
    gradient_f -- noted as g
    approx_hess_f -- noted as B
    ...
    regularization term is noted as nu
    modification factor of regularization term is noted as gamma
    ok_ratio -- noted as eta, in the range [0, 1/4)
    iterations -- 

    some Codes notations from the Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization by Yuval Tassa, Tom Erez and Emanuel Todorov.
    """

    f, g, B = fun, gradient_f, approx_hessian_f
    nu_k, gamma_k = initial_regularization, initial_modification_factor
    nu_min, gamma_min = minimal_regularization, minimal_modification_factor
    x_k = x0

    n = np.size(x_k)

    def increase_nu():
        nonlocal nu_k, gamma_k
        gamma_k = max(gamma_min, gamma_k * gamma_min)
        nu_k = max(nu_min, gamma_k * nu_k)

    def decrease_nu():
        nonlocal nu_k, gamma_k
        gamma_k = min(1/gamma_min, gamma_k/gamma_min)
        nu = gamma_k * nu_k
        nu_k = nu if nu > nu_min else 0 
    
    for k in range(iterations):
        g_k = g(x_k)
        grad_norm = np.linalg.norm(g_k)
        if grad_norm < 1e-10:
            return x_k

        f_k = f(x_k)
        B_k = B(x_k)    
        H_k = B_k + nu_k * np.identity(n)

        p_k = solve_subproblem(f_k, g_k, B_k, nu_k)
        actual_reduction = f_k - f(x_k + p_k)
        predicted_reduction = -(g_k @ p_k + 1/2 * p_k @ H_k @ p_k)     #   
        ro_k =  actual_reduction / predicted_reduction

        if ro_k < 1/4:
            increase_nu()
        elif ro_k > 3/4:
            decrease_nu()
        
        if ro_k > ok_ratio:
            x_k = x_k + p_k

    return x_k


def regularization_plot(fun,
                   x0: np.array,
                   gradient_f,
                   approx_hessian_f,
                   iterations: int,
                   minimal_regularization: float = 1e-6,
                   initial_regularization: float = 1,
                   minimal_modification_factor: float = 2,
                   initial_modification_factor: float = 2,
                   ok_ratio: float = 0.2) -> "x*":
    """Compute the local minimum by Regularization method. 

    Arguments:
    fun -- the function to be optimized
    x0 -- the initial feasible starting point
    gradient_f -- noted as g
    approx_hess_f -- noted as B
    ...
    regularization term is noted as nu
    modification factor of regularization term is noted as gamma
    ok_ratio -- noted as eta, in the range [0, 1/4)
    iterations -- 

    some Codes notations from the Synthesis and Stabilization of Complex Behaviors through
Online Trajectory Optimization by Yuval Tassa, Tom Erez and Emanuel Todorov.
    """

    f, g, B = fun, gradient_f, approx_hessian_f
    nu_k, gamma_k = initial_regularization, initial_modification_factor
    nu_min, gamma_min = minimal_regularization, minimal_modification_factor
    x_k = x0

    f_values = []
    #p_values =[]
    ro_values = []
    nu_values =[initial_regularization]
    grad_norms = []
    predicted_reductions = []
    actual_reductions = []

    n = np.size(x_k)

    def increase_nu():
        nonlocal nu_k, gamma_k
        gamma_k = max(gamma_min, gamma_k * gamma_min)
        nu_k = max(nu_min, gamma_k * nu_k)

    def decrease_nu():
        nonlocal nu_k, gamma_k
        gamma_k = min(1/gamma_min, gamma_k/gamma_min)
        nu = gamma_k * nu_k
        nu_k = nu if nu > nu_min else 0 
    
    for k in range(iterations):
        g_k = g(x_k)
        grad_norm = np.linalg.norm(g_k)
        if grad_norm < 1e-10:
            break

        f_k = f(x_k)
        B_k = B(x_k)    
        H_k = B_k + nu_k * np.identity(n)

        p_k = solve_subproblem(f_k, g_k, B_k, nu_k)
        actual_reduction = f_k - f(x_k + p_k)
        predicted_reduction = -(g_k @ p_k + 1/2 * p_k @ H_k @ p_k)     #   
        ro_k =  actual_reduction / predicted_reduction


        if ro_k < 1/4:
            increase_nu()
        elif ro_k > 3/4:
            decrease_nu()
        
        if ro_k > ok_ratio:
            x_k = x_k + p_k

        f_values.append(f_k)
        grad_norms.append(grad_norm)
        actual_reductions.append(actual_reduction)
        predicted_reductions.append(predicted_reduction)
        ro_values.append(ro_k)
        nu_values.append(nu_k)


    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    # ax5 = fig.add_subplot(225)

    ax1.title.set_text("reductions")
    ax1.plot(predicted_reductions, '-b', label = "predicted reductions")
    ax1.plot(actual_reductions, '--r', label = "actual reductions")
    ax1.legend()
    ax2.title.set_text("function values")
    ax2.plot(f_values)
    #ax3.set_ylabel("f(x_k)")
    ax3.title.set_text("gradient norms")
    ax3.semilogy(grad_norms)
    ax4.title.set_text("Regularization terms evolution")
    ax4.plot(nu_values)

    fig.suptitle("Starting point: {}".format(x0))

    plt.show()

    return x_k