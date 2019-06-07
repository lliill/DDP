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

_F = Matrix([F1, F2, F3, F4])

_F_u = _F.jacobian(U)

_F_x = _F.jacobian(X)

_F_xx = [hessian(Fi, X) for Fi in _F]





def F(x, u): #u must be 1-D np array, like np.array((1,)), not np.array(1)
    aux = lambdify((X, U), _F, "numpy")
    return np.squeeze(aux(x, u))


def F_u(x, u):
    aux = lambdify((X, U), _F_u, "numpy")
    return np.squeeze(aux(x, u))

F_x = lambdify((X, U), _F_x, "numpy")


def F_xx(x,u):
    aux = lambdify((X, U), _F_xx, "numpy")
    return np.array(aux(x,u))

def F_xu(x,u):
    aux = F.diff(X).diff(U) #F_ux
    aux = lambdify((X, U), aux, "numpy")
    res = np.squeeze(np.array(aux(x, u))).T #F_xu
    return res

# def F(states_controls: np.array) -> "states":
#     x, theta, v, w, f = states_controls # w is the velocity of theta

#------------------------------------------------------------------
#Loss design
_L =   (theta - pi)**2 + w**2 + f**2
_L_T = (theta - pi)**2 + w**2

_L_x = _L.diff(X)

_L_u = _L.diff(U)

_L_T_x = _L_T.diff(X)

def L_x(x, u):
    aux = lambdify((X, U), _L_x, "numpy")
    #not completed