import numpy as np
from newton_inequality import solve_TR_subproblem 
import time

# def m(p) : return 1 + (-2*np.ones(3)) @ p + p @ p




if __name__ == '__main__':
    f = 1
    g = -2 * np.ones(3)
    B = 2 * np.eye(3)

    Delta = 0.5

    begin = time.process_time()
    sol = solve_TR_subproblem(f, g, B, Delta)
    end = time.process_time()
    elapsed_time = end - begin 

    print(sol, "with", elapsed_time, "seconds")