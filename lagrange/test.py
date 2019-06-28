import numpy as np
# from numpy import array, dot
# from qpsolvers import solve_qp

# M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
# P = dot(M.T, M)  # quick way to build a symmetric matrix
# q = dot(array([3., 2., 3.]), M).reshape((3,))
# G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
# h = array([3., 2., -2.]).reshape((3,))
# A = array([1., 1., 1.])
# b = array([1.])

# print ("QP solution:", solve_qp(P, q, A = A, b = b))
def test_cvxopt():
    from cvxopt import matrix, solvers
    Q = 2*matrix([ [2, .5], [.5, 1] ])
    p = matrix([1.0, 1.0])
    A = matrix([1.0, 1.0], (1,2))
    b = matrix(1.0)
    # A = matrix([[1,1],
    #             [0,0]])
    # b = matrix([1,0])
    sol=solvers.qp(Q, p, A = A,b = b)

    print(sol['x'])


test_cvxopt()


from DDP.lagrange.equality import augmented_lagrange_equality


def test_LA():
    Q = np.array([[2, 0.5],
                  [0.5, 1]])
    p = np.array([1, 1])

    def f(x):
        x1 = x[0]
        x2 = x[1]
        return 2*x1**2 + x2**2 + x1*x2 + x1 + x2

    def Diffs_f(x):
        return 2*Q @ x + p, 2*Q       
    

    def c(x):
        x1 = x[0]
        x2 = x[1]
        return x1 + x2 - 1    

    def Diffs_c(x):
        return np.array([1, 1]), np.zeros((2,2))   

    C = [c]         
    Diffs_C = [Diffs_c]

    x = augmented_lagrange_equality(f, C, Diffs_f, Diffs_C, np.array([0, 0]), np.array([1]))
    print(x)

test_LA()