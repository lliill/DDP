import numpy as np
from trust_region import *
import time

def test_TR_plot():
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
    sol = trust_region_plot( f, x0, grad_f, hess_f, 30)
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
    sol = trust_region(f, x0, grad_f, hess_f, 30)
    end = time.process_time()
    elapsed_time = end - begin

    print(sol, "with", elapsed_time, "seconds") 

def test_subprob():
    f = 11
    g = np.array([-2, -20])
    B = np.array([[42, 0],
                  [0, 20]])

def test_subprob2():
    # def m(p) : return 1 + (-2*np.ones(3)) @ p + p @ p
    f = 0.9071964022521671
    g = np.array([-1.89830285, -0.06840818])
    B = np.array( [[ 2.31802258, -1.90371335],
                    [-1.90371335, 20.        ]])

    Delta = 2

    begin = time.process_time()
    sol = solve_TR_subproblem(f, g, B, Delta)
    end = time.process_time()
    elapsed_time = end - begin 

    print(sol, "with", elapsed_time, "seconds")  

if __name__ == '__main__':
    # test_subprob2()
    # test_TR()
    # test_subprob()
    test_TR_plot()