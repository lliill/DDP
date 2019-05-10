# Trust region notation from Numerical Optimization of J.Nocedal
import numpy as np
import matplotlib.pyplot as plt
from math import isclose
from newton_barriers import barrier_inegality

def solve_TR_subproblem(f: "f(x_k)",
                        g: "Grad f(x_k)",
                        B: "B_k symmetic and uniformly bounded, Aprox Hess of f(x_k)",
                        Delta: "Delta_k > 0, Trust Region radius") -> "p_k":
    """Solve min_p m(p) = f_k + g_k^T p + 1/2 p^T B_k p
                s.t. ||p|| <= Delta_k.
        the type of g needs to support @ operation"""

    def difficult_case():
        n = np.size(g)
        p0 = np.zeros(n)
            
        def m(p): return f + g @ p + 1/2 * p @ B @ p

        def grad_m(p): return g + B @ p

        def hess_m(p): return B

        def c(p): 
            """Constraint function of trust region."""
            return p @ p - Delta**2

        def grad_c(p): return 2 * p

        def hess_c(p): return 2 * np.eye(n)

        return barrier_inegality([m, c], p0, 1e-6, [grad_m, grad_c], [hess_m, hess_c])
    
    try:
        B_inv = np.linalg.inv(B) #if B is positive definite and symetric
        sol = -B_inv @ g
        if np.linalg.norm(sol) <= Delta:
            return sol
        else:
            return difficult_case()

    except np.linalg.LinAlgError:

        return difficult_case()


def trust_region(f, x_0 :np.array,
                gradient_f,
                approx_hess_f,
                iterations: int,
                max_radius: float = 1,
                init_radius: float = 1,
                ok_range: float = 0.2,
                ) -> "x*":
    """Compute the local minimum by Trust Region method with a plot. 

    Arguments:
    f -- the function to be optimized
    x0 -- the initial feasible starting point
    gradient_f -- noted as g
    approx_hess_f -- noted as B
    ...
    init_radius -- noted as Delta_k, in the range (0, max_radius)
    ok_range -- noted as eta, in the range [0, 1/4)
    iterations -- 

    Codes notations from the Numerical Optimization of J.Nocedal, Chapter 4, page 68, 69.
    """  

    g, B, Delta_k, eta = gradient_f, approx_hess_f, init_radius, ok_range 

    x_k = x_0

    predicted_reductions = []
    actual_reductions = [] #
    for k in range(iterations):
        f_k = f(x_k)
        g_k = g(x_k)
        B_k = B(x_k)

        p_k  = solve_TR_subproblem(f_k, g_k, B_k, Delta_k)
        actual_reduction = f_k - f(x_k + p_k)
        predicted_reduction = -(g_k @ p_k + 1/2 * p_k @ B_k @ p_k)
        ro_k =  actual_reduction / predicted_reduction

  
        grad_norm = np.linalg.norm(g_k)

        if isclose(grad_norm, 0, rel_tol= 0.02):
            return x_k

        if ro_k < 1/4:
            Delta_k = 1/4 * Delta_k
        else:
            if ro_k > 3/4 and isclose(np.linalg.norm(p_k), Delta_k, rel_tol=0.05):#np.linalg.norm(p_k) == Delta_k
                Delta_k = min(2*Delta_k, max_radius)
        if ro_k > eta:
            x_k = x_k + p_k
           
    return x_k

def trust_region_plot(f, x_0 :np.array,
                gradient_f,
                approx_hess_f,
                iterations: int,
                max_radius: float = 1,
                init_radius: float = 1,
                ok_range: float = 0.2,
                ) -> "x*":
    """Compute the local minimum by Trust Region method with a plot. 

    Arguments:
    f -- the function to be optimized
    x0 -- the initial feasible starting point
    gradient_f -- noted as g
    approx_hess_f -- noted as B
    ...
    init_radius -- noted as Delta_k, in the range (0, max_radius)
    ok_range -- noted as eta, in the range [0, 1/4)
    iterations -- 

    Codes notations from the Numerical Optimization of J.Nocedal, Chapter 4, page 68, 69.
    """  

    g, B, Delta_k, eta = gradient_f, approx_hess_f, init_radius, ok_range 

    x_k = x_0
    x = [x_k]
    f_values = []
    p =[]
    ro = []
    Delta = [Delta_k]
    Grad_norms = []
    Hess = []

    predicted_reductions = []
    actual_reductions = [] #
    for k in range(iterations):
        f_k = f(x_k)
        g_k = g(x_k)
        grad_norm = np.linalg.norm(g_k)
        if isclose(grad_norm, 0, rel_tol= 0.02):
            break
        B_k = B(x_k)

        p_k  = solve_TR_subproblem(f_k, g_k, B_k, Delta_k)
        actual_reduction = f_k - f(x_k + p_k)
        predicted_reduction = -(g_k @ p_k + 1/2 * p_k @ B_k @ p_k)
        ro_k =  actual_reduction / predicted_reduction

        actual_reductions.append(actual_reduction)
        predicted_reductions.append(predicted_reduction)#

        f_values.append(f_k)
        Grad_norms.append(np.linalg.norm(g_k))
        # grad.append(g_k)#
        p.append(p_k)
        ro.append(ro_k)
        Hess.append(B_k)#

        if ro_k < 1/4:
            Delta_k = 1/4 * Delta_k
        else:
            if ro_k > 3/4 and isclose(np.linalg.norm(p_k), Delta_k, rel_tol=0.05):#np.linalg.norm(p_k) == Delta_k
                Delta_k = min(2*Delta_k, max_radius)
        if ro_k > eta:
            x_k = x_k + p_k

        #k+1
        Delta.append(Delta_k)
        x.append(x_k)

    # plt.figure(1, figsize=(9, 3))
    # plt.subplot(131)
    # plt.plot(f, range(iterations))
    # plt.ylabel("f(x_k)")
    # plt.subplot(132)
    # plt.plot(ro, range(iterations))
    # plt.ylabel("ro_k")
    # plt.subplot(133)
    # plt.plot(Delta, range(iterations+1))
    # plt.ylabel("Delta_k")
    # plt.suptitle('Trust Region Evolution Plotting')
    # plt.show()

    # print(f_values[1], "f_k \n", grad[1], "g_k\n", Hess[1], "B_k\n")
    # print(p[1], "p")
    # print(decr_m[1])

    # fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    # ax1.title.set_text('f_k Plot')
    # ax2.title.set_text('gradient norm Plot')
    # ax3.title.set_text('Decreasing Ratio Plot')
    # ax4.title.set_text('Trust Region Plot')
    # ax1.plot(range(iterations), f_values)
    # ax1.set_ylabel("f(x_k)")
    # #ax2.plot(range(iterations), grad)
    # ax2.set_ylabel("gradient f(x_k) norm")
    # ax3.plot(range(iterations), ro)
    # ax3.set_ylabel("ro_k")
    # ax4.plot(range(iterations+1), Delta)
    # ax4.set_ylabel("Delta_k")
    # plt.show()


    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(151)
    ax2 = fig2.add_subplot(152)
    ax3 = fig2.add_subplot(153)
    ax4 = fig2.add_subplot(154)
    ax5 = fig2.add_subplot(155)

    ax1.title.set_text("predicted reductions")
    ax1.plot(predicted_reductions)
    ax2.title.set_text("actual reductions")
    ax2.plot(actual_reductions)
    ax3.title.set_text("f(x_k)")
    ax3.plot(f_values)
    ax3.set_ylabel("f(x_k)")
    ax4.title.set_text("gradient norm")
    ax4.plot(Grad_norms)
    ax5.title.set_text("Trust Region Radius")
    ax5.plot(Delta)

    plt.show()

           
    return x_k
