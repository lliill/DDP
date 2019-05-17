import numpy as np
from regularization import *
import time

def test_plot():
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

    x0 = np.array([0, 5])  

    begin = time.process_time()
    sol = regularization_plot(f, x0, grad_f, hess_f, 30)
    end = time.process_time()
    elapsed_time = end - begin

    print("optimum is", sol, "with", elapsed_time, "seconds")    

def test():
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
    sol = regularization(f, x0, grad_f, hess_f, 30)
    end = time.process_time()
    elapsed_time = end - begin

    print(sol, "with", elapsed_time, "seconds") 

if __name__ == '__main__':
    test_plot() 
    # test()