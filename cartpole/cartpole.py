from sympy import *
import numpy as np

# the same configurations as gym cartpole
M_p = 0.1 # masscart 
M_c = 1 # masspole
L = 1
G = 9.8

x, theta, v, w, f = symbols("x, theta, v, w, f")
X = Array([x, theta, v, w])
U = Array([f])

# I hope it will be the same step as the gym one
F1 = v
F2 = w
F3 = (f + M_p*sin(theta)*(G*cos(theta) + L*w**2) ) / (M_c + M_p * sin(theta)**2)
F4 = - (f*cos(theta) + (M_p + M_c)*G*sin(theta) + L*M_p*w**2*sin(theta)*cos(theta) ) / (L*(M_c + M_p*sin(theta)**2) )

F = Matrix([F1, F2, F3, F4])

F_u = F.jacobian(U)

F_x = F.jacobian(X)

F_xx = [hessian(Fi, X) for Fi in F]





def F(x, u): #u must be 1-D np array, like np.array((1,)), not np.array(1)
    aux = lambdify((X, U), F, "numpy")
    return np.squeeze(aux(x, u))


def F_u(x, u):
    aux = lambdify((X, U), F_u, "numpy")
    return np.squeeze(aux(x, u))

F_x = lambdify((X, U), F_x, "numpy")


def F_xx(x,u):
    aux = lambdify((X, U), F_xx, "numpy")
    return np.array(aux(x,u))

def F_xu(x,u):
    aux = F.diff(X).diff(U)
    aux = lambdify((X, U), aux, "numpy")
    res = np.squeeze(np.array(aux(x, u))).T
    return res

# def F(states_controls: np.array) -> "states":
#     x, theta, v, w, f = states_controls # w is the velocity of theta