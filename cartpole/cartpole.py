from sympy import *


M_p = 1
M_c = 10
L = 1
G = 9.80665

x, theta, v, w, f = symbols("x, theta, v, w, f")
X = Matrix([x, theta])
U = f

F1 = v
F2 = w
F3 = (f + M_p*sin(theta)*(G*cos(theta) + L*w**2) ) / (M_c + M_p * sin(theta)**2)
F4 = - (f*cos(theta) + (M_p + M_c)*G*sin(theta) + L*M_p*w**2*sin(theta)*cos(theta) ) / (L*(M_c + M_p*sin(theta)**2)


# def F(states_controls: np.array) -> "states":
#     x, theta, v, w, f = states_controls # w is the velocity of theta