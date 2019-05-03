#Newton Notations in the Convex Optimization of S.Boyd
# Trust region notation from Numerical Optimization of J.Nocedal
import numpy as np
import math
def log(x):
    try:
        return math.log(x)
    except ValueError:
        return math.inf


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
        Hess_f_x_inv = np.linalg.inv(hessian_f(x))
        
        decrement = Grad_f_x @ Hess_f_x_inv @ Grad_f_x
        if decrement/2 <= e:
            return x
        newton_step = -Hess_f_x_inv * Grad_f_x

        t = line_search(f, x, newton_step, Grad_f_x, a, b)
        x = x + t*newton_step

def inegality_constraints(
        f0: "convex function to be minimized",
        fi_s: "list of convex inegality constraints",
        x:  "strictly feasible starting point",
        e:  "tolerance, >0",
        grad_f,
        hess_f,
        t:  "t0 > 0" = 1,
        nu: "> 1" = 5) -> "x*":

    n = x.size
    m = len(fi_s)
    grad_f0 = grad_f[0]
    hess_f0 = hess_f[0]
    grad_fi_s = grad_f[1:]
    hess_fi_s = hess_f[1:]
    def phi(x): return -sum(log(-f_i(x)) for f_i in fi_s)
    def grad_phi(x):
        return sum(-1/(fi(x)) * grad_fi(x) for grad_fi in grad_fi_s)
    def hess_phi(x):
        l = []
        for i in range(m):
            fi_x = fi_s[i](x)
            hess_fi_x = hess_fi_s[i](x)
            grad_fi_x = grad_fi_s[i](x)

            part1 = hess_fi_x/fi_x

            _l = [grad_fi_x[j]*grad_fi_x[k] for j in range(n) for k in range(n)]
            part2 = np.reshape(_l, (n,n)) / fi_x**2

            l.append(part1 + part2)
        return sum(l)
    
    while True:
        
        x_star_t = newton_method(lambda x: t * f0(x) + phi(x),
                                 x, e,
                                 lambda x: t* grad_f0(x) + grad_phi(x),
                                 lambda x: t* hess_f0(x) + hess_phi(x))
        x = x_star_t
        if m/t < e:
            return x
        t = nu * t

def solve_TR_subproblem(f: "f(x_k)",
                        g: "Grad f(x_k)",
                        B: "B_k symmetic and uniformly bounded, Aprox Hess of f(x_k)",
                        Delta: "Delta_k > 0, Trust Region radius") -> "p_k":
    """Solve min_p m(p) = f_k + g_k^T p + 1/2 p^T B_k p
                s.t. ||p|| <= Delta_k"""
    try:
        B_inv = np.linalg.inv(B) #if B is positive definite and symetric
        sol = B_inv @ g
        if np.linalg.norm(sol) <= Delta:
            return sol
    except np.linalg.LinAlgError:
            
        def m(p): return f + g @ p + 1/2 * p @ B @ p
    
        
