import numpy as np
from newton_inequality import *
import time



def test_subprob():
    # def m(p) : return 1 + (-2*np.ones(3)) @ p + p @ p
    f = 1
    g = -2 * np.ones(3)
    B = 2 * np.eye(3)

    Delta = 0.5

    begin = time.process_time()
    sol = solve_TR_subproblem(f, g, B, Delta)
    end = time.process_time()
    elapsed_time = end - begin 

    print(sol, "with", elapsed_time, "seconds")    


def test_TR():
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return 10*(x2-x1**2)**2 + (1-x1)**2
    
    def grad_f(x):
        x1 = x[0]
        x2 = x[1]
        return np.array([40*x1**3 - 40*x1*x2 + 2*x1 -2, 
                        20*(x2-x1**2)])

    def hess_f(x):
        x1 = x[0]
        x2 = x[1]
        return np.array([
            [120*x1**2 - 40*x2 +2, -40*x1],
            [-40*x1,                20]
        ])

    x0 = np.array([0, -1])

    begin = time.process_time()
    sol = trust_region_plot(f, x0, grad_f, hess_f, 2, 1, 0.2, 50)
    end = time.process_time()
    elapsed_time = end - begin

    print(sol, "with", elapsed_time, "seconds") 

def test_subprob_plot():
    # def m(p) : return 1 + (-2*np.ones(3)) @ p + p @ p
    f = 0.9071964022521671
    g = np.array([-1.89830285, -0.06840818])
    B = np.array( [[ 2.31802258, -1.90371335],
 [-1.90371335, 20.        ]])

    Delta = 0.5

    begin = time.process_time()
    sol = solve_TR_subproblem_plot(f, g, B, Delta)
    end = time.process_time()
    elapsed_time = end - begin 

    print(sol, "with", elapsed_time, "seconds")  

if __name__ == '__main__':
    test_TR()