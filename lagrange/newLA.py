import numpy as np
import matplotlib.pyplot as plt
from test_graph import newton_method
from functools import partial
# from time import time
from scipy import linalg

Buffers = {'r': None, 
           'Df': None,
           'Dc': None}

def optimize(f,
            gradient_f,
            hessian_f, 
            c : "x |-> array", 
            jacobian_c,
            hessian_c,
            x :"feasible initial",
            constraints_size : int,
            stopping_criteria: "for ||grad L|| and ||c(x)||" = {"grad_L" :1e-6,
                                                'constraints_residuals': 1e-6},  
            penalization_increase_rate = 2, 
            initial_residuals_tolerance = 1e-6,                                              
            max_iter = 200,
            max_penalization = 1e3,
            primal_sol = None,
            dual_sol = None):

    def augemented_Lagrangian(lamb: np.array, mu: float, x):
        C_x = c(x)
        return f(x) + lamb @ C_x + mu/2 * np.sum(C_x**2) 

    def diffs_LA(lamb, mu, x):
        grad_f_x, hess_f_x = gradient_f(x), hessian_f(x)
        A, H = jacobian_c(x), hessian_c(x)
        r = c(x)
        Buffers['r'], Buffers['Df'], Buffers['Dc'] = r, grad_f_x, A

        grad_x_LA = grad_f_x + (lamb + mu * r) @ A
        hess_x_LA = hess_f_x + mu * A.T @ A + np.tensordot(lamb + mu * r, H, axes = 1)

        return (grad_x_LA, hess_x_LA)

    def norm_inf(x):
        return linalg.norm(x, ord = np.inf)

    def KKT_test(grad_L_norm, constraints_residuals_norm):
        Conditions = []
        Conditions.append( grad_L_norm < stopping_criteria['grad_L'] )
        Conditions.append( constraints_residuals_norm < stopping_criteria['constraints_residuals'])
        return all(Conditions)

    def update(penalization, constraints_residuals, residual_tolerance) -> "mu, tau":
        if norm_inf(constraints_residuals) < residual_tolerance:
            return penalization, residual_tolerance/penalization**0.9
        else:
            new_penalization = penalization_increase_rate * penalization               
            if new_penalization > max_penalization:
                return penalization, 1/penalization**0.1
            else:
                return new_penalization, 1/new_penalization**0.1

    mu = 2
    lamb = np.zeros(constraints_size)
    residuals_tolerance = initial_residuals_tolerance

    Distances_to_optimum =[]
    Residuals_Norm = []
    Grad_L_Norm = []
    Dual_Distances_to_optimum = []
    Penalizations = []
    Residuals_tolerances = []

    for _ in range(max_iter):
        print(_)
        L_A = partial(augemented_Lagrangian, lamb, mu)
        DL_A = partial(diffs_LA, lamb, mu)
        x = newton_method(L_A, DL_A, x, 1e-8)        

        r = Buffers['r']
        gradient_Lagrangian = Buffers['Df'] + lamb @ Buffers['Dc']

        grad_L_norm = norm_inf(gradient_Lagrangian)
        residuals_norm = norm_inf(r)

        Distances_to_optimum.append(norm_inf(x - primal_sol))
        Dual_Distances_to_optimum.append(norm_inf(lamb - dual_sol))
        Residuals_Norm.append(residuals_norm)
        Grad_L_Norm.append(grad_L_norm)
        Penalizations.append(mu)
        Residuals_tolerances.append(residuals_tolerance)

        if residuals_norm <= residuals_tolerance:
            if KKT_test(grad_L_norm, residuals_norm):
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233)
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236)

                ax1.title.set_text("||x - x*||_inf")
                ax2.title.set_text("||c(x)||_inf")
                ax3.title.set_text("||gradient Lagrangian||_inf")
                ax4.title.set_text("||lambda - lambda*||_inf")
                ax5.title.set_text("Penalizations")
                ax6.title.set_text("Residuals_tolerances")

                ax1.plot(Distances_to_optimum)
                ax2.plot(Residuals_Norm)
                ax3.plot(Grad_L_Norm)
                ax4.plot(Dual_Distances_to_optimum)
                ax5.plot(Penalizations)
                ax6.plot(Residuals_tolerances)

                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax3.set_yscale('log')
                ax4.set_yscale('log')
                # ax5.set_yscale('log')
                ax6.set_yscale('log')
                plt.show()
                return x, 'done'

        lamb = lamb + mu * r
        mu, residuals_tolerance = update(mu, r ,residuals_tolerance)

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
    ax5.title.set_text("Penalizations")
    ax6.title.set_text("Residuals_tolerances")

    ax1.plot(Distances_to_optimum)
    ax2.plot(Residuals_Norm)
    ax3.plot(Grad_L_Norm)
    ax4.plot(Dual_Distances_to_optimum)
    ax5.plot(Penalizations)
    ax6.plot(Residuals_tolerances)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    # ax5.set_yscale('log')
    ax6.set_yscale('log')
    plt.show()
    return x, 'not done'    






