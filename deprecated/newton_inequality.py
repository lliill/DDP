#Newton Notations in the Convex Optimization of S.Boyd
# Trust region notation from Numerical Optimization of J.Nocedal
import numpy as np
import matplotlib.pyplot as plt
import math
from functools import partial
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

def newton_method(
        f: "convex function to be minimized",
        x: "starting point in the feasible domain of f",
        e: "tolerance, >0",
        gradient_f,
        hessian_f,
        a: "line search parameter, affinity of approximation" = 0.25,
        b: "line search parameter, decreasing rate" = 0.5) -> "x*":
    """Newton's method in the page 487"""
    while True:
        Grad_f_x = gradient_f(x)
        # hfx = hessian_f(x)
        Hess_f_x_inv = np.linalg.inv(hessian_f(x))
        
        decrement = Grad_f_x @ Hess_f_x_inv @ Grad_f_x
        if decrement/2 <= e:
            return x
        newton_step = -Hess_f_x_inv @ Grad_f_x

        t = line_search(f, x, newton_step, Grad_f_x, a, b)
        x = x + t*newton_step

def inegality_constraints(
        f_s: "list of convex function to be minimized and convex inegality constraints",
        x:  "strictly feasible starting point",
        e:  "tolerance, >0",
        grad_f_s: "list of gradients from f0 to fm",
        hess_f_s: "list of hessians from f0 to fm",
        t:  "t0 > 0" = 1,
        nu: "> 1" = 5) -> "x*":

    n = x.size
    f0, constraints = f_s[0], f_s[1:]
    m = len(constraints)
    grad_f0, grad_constraints = grad_f_s[0], grad_f_s[1:]
    hess_f0, hess_constraints = hess_f_s[0], hess_f_s[1:]

    # _ = -constraints[0](x)

    def phi(x): 
        # print(x)
        # for f_i in constraints:
        #     print(f_i(x))
        
        return -sum(log(-f_i(x)) for f_i in constraints)
    def grad_phi(x):
        return sum(-1/(constraints[i](x)) * grad_constraints[i](x) for i in range(m))
    def hess_phi(x):
        l = []
        for i in range(m):
            fi_x = constraints[i](x)
            hess_fi_x = hess_constraints[i](x)
            grad_fi_x = grad_constraints[i](x)

            part1 = -hess_fi_x/fi_x

            _l = [grad_fi_x[j]*grad_fi_x[k] for j in range(n) for k in range(n)]
            part2 = np.reshape(_l, (n,n)) / fi_x**2

            l.append(part1 + part2)
        return sum(l)
    
    while True:
        def f(t, x): return t * f0(x) + phi(x)
        def grad_f(t, x): return t* grad_f0(x) + grad_phi(x)
        def hess_f(t, x): return t* hess_f0(x) + hess_phi(x)
        x_star_t = newton_method(partial(f, t),
                                 x, e,
                                 partial(grad_f, t),
                                 partial(hess_f, t))
        # x_star_t = newton_method(lambda x: t * f0(x) + phi(x),
        #                          x, e,
        #                          lambda x: t* grad_f0(x) + grad_phi(x),
        #                          lambda x: t* hess_f0(x) + hess_phi(x))
        x = x_star_t
        if m/t < e:
            return x
        t = nu * t

def barrier_inegality_plot(
        f_s: "list of convex function to be minimized and convex inegality constraints",
        x:  "strictly feasible starting point",
        e:  "tolerance, >0",
        grad_f_s: "list of gradients from f0 to fm",
        hess_f_s: "list of hessians from f0 to fm",
        t:  "t0 > 0" = 1,
        nu: "> 1" = 5) -> "x*":

    n = x.size
    f0, constraints = f_s[0], f_s[1:]
    m = len(constraints)
    grad_f0, grad_constraints = grad_f_s[0], grad_f_s[1:]
    hess_f0, hess_constraints = hess_f_s[0], hess_f_s[1:]

    # _ = -constraints[0](x)

    def phi(x): 
        # print(x)
        # for f_i in constraints:
        #     print(f_i(x))
        
        return -sum(log(-f_i(x)) for f_i in constraints)
    def grad_phi(x):
        return sum(-1/(constraints[i](x)) * grad_constraints[i](x) for i in range(m))
    def hess_phi(x):
        l = []
        for i in range(m):
            fi_x = constraints[i](x)
            hess_fi_x = hess_constraints[i](x)
            grad_fi_x = grad_constraints[i](x)

            part1 = -hess_fi_x/fi_x

            _l = [grad_fi_x[j]*grad_fi_x[k] for j in range(n) for k in range(n)]
            part2 = np.reshape(_l, (n,n)) / fi_x**2

            l.append(part1 + part2)
        return sum(l)
    
    f0_values = [f0(x)]

    while True:
        def f(t, x): return t * f0(x) + phi(x)
        def grad_f(t, x): return t* grad_f0(x) + grad_phi(x)
        def hess_f(t, x): return t* hess_f0(x) + hess_phi(x)
        x_star_t = newton_method(partial(f, t),
                                 x, e,
                                 partial(grad_f, t),
                                 partial(hess_f, t))
        # x_star_t = newton_method(lambda x: t * f0(x) + phi(x),
        #                          x, e,
        #                          lambda x: t* grad_f0(x) + grad_phi(x),
        #                          lambda x: t* hess_f0(x) + hess_phi(x))

        x = x_star_t
        f0_values.append(f0(x))
        if m/t < e:
            plt.plot(f0_values)
            plt.show()
            print(f0_values[-1], "f(x)*")
            print(x, 'x*')
            return x
        t = nu * t


def solve_TR_subproblem_plot(f: "f(x_k)",
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

        return barrier_inegality_plot([m, c], p0, 1e-6, [grad_m, grad_c], [hess_m, hess_c])

    
        
    try:
        B_inv = np.linalg.inv(B) #if B is positive definite and symetric
        sol = B_inv @ g
        if np.linalg.norm(sol) <= Delta:
            return sol
        else:
            return difficult_case()

    except np.linalg.LinAlgError:

        return difficult_case()


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

        return inegality_constraints([m, c], p0, 1e-6, [grad_m, grad_c], [hess_m, hess_c])

    
        
    try:
        B_inv = np.linalg.inv(B) #if B is positive definite and symetric
        sol = B_inv @ g
        if np.linalg.norm(sol) <= Delta:
            return sol
        else:
            return difficult_case()

    except np.linalg.LinAlgError:

        return difficult_case()

        
def trust_region(f, x_0 :np.array,
                gradient_f,
                approx_hess_f,
                max_radius: float,
                init_radius: float,
                ok_range: float,
                iterations: int) -> "x*":
    """Compute the local minimum by Trust Region method. 

    Arguments:
    f -- the function to be optimized
    x0 -- the initial feasible starting point
    gradient_f -- noted as g
    approx_hess_f -- noted as B
    ...
    init_radius -- noted as Delta, in the range (0, max_radius)
    ok_range -- noted as eta, in the range [0, 1/4)
    iterations -- 

    Codes notations from the Numerical Optimization of J.Nocedal, Chapter 4, page 68, 69.
    """  

    g, B, Delta, eta = gradient_f, approx_hess_f, init_radius, ok_range 

    x_k = x_0
    for k in range(iterations):
        f_k = f(x_k)
        g_k = g(x_k)
        B_k = B(x_k)

        p_k  = solve_TR_subproblem(f_k, g_k, B_k, Delta)
        ro_k = - (f_k - f(x_k + p_k)) / (g_k @ p_k + 1/2 * p_k @ B_k @ p_k)

        if ro_k < 1/4:
            Delta = 1/4 * Delta
        else:
            if ro_k > 3/4 and math.isclose(np.linalg.norm(p_k), Delta, rel_tol=0.05):#np.linalg.norm(p_k) == Delta
                Delta = min(2*Delta, max_radius)
        if ro_k > eta:
            x_k = x_k + p_k

    return x_k

def trust_region_plot(f, x_0 :np.array,
                gradient_f,
                approx_hess_f,
                max_radius: float,
                init_radius: float,
                ok_range: float,
                iterations: int) -> "x*":
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
    grad = []
    Hess = []

    decr_m = []#
    for k in range(iterations):
        f_k = f(x_k)
        g_k = g(x_k)
        B_k = B(x_k)

        p_k  = solve_TR_subproblem(f_k, g_k, B_k, Delta_k)
        actual_reduction = f_k - f(x_k + p_k)
        predicted_reduction = -(g_k @ p_k + 1/2 * p_k @ B_k @ p_k)
        ro_k =  actual_reduction / predicted_reduction
        decr_m.append(predicted_reduction)#

        f_values.append(f_k)
        #grad.append(np.linalg.norm(g_k))
        grad.append(g_k)#
        p.append(p_k)
        ro.append(ro_k)
        Hess.append(B_k)#

        if ro_k < 1/4:
            Delta_k = 1/4 * Delta_k
        else:
            if ro_k > 3/4 and math.isclose(np.linalg.norm(p_k), Delta_k, rel_tol=0.05):#np.linalg.norm(p_k) == Delta_k
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

    print(f_values[1], "f_k \n", grad[1], "g_k\n", Hess[1], "B_k\n")
    print(p[1], "p")
    print(decr_m[1])
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

    # fig2 = plt.figure(2)
    # plt.plot(range(iterations), decr_m)
    # plt.show()

           
    return x_k