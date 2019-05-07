#Newton Notations in the Convex Optimization of S.Boyd
# Trust region notation from Numerical Optimization of J.Nocedal
import numpy as np
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

        
        
