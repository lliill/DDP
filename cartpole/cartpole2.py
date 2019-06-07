import numpy as np
from math import sin, cos

# the same configurations as gym cartpole
M_p = 0.1 # masscart 
M_c = 1 # masspole
L = 1
G = 9.8

def F(x, u): #NO!!!!! pas seulement la derivee!!!!
    x, theta, v, w = x
    f, = u

    # I hope it will be the same step as the gym one
    F1 = v
    F2 = w
    F3 = (f + M_p*sin(theta)*(G*cos(theta) + L*w**2) ) / (M_c + M_p * sin(theta)**2)
    F4 = - (f*cos(theta) + (M_p + M_c)*G*sin(theta) + L*M_p*w**2*sin(theta)*cos(theta) ) / (L*(M_c + M_p*sin(theta)**2) )

    return np.array([F1, F2, F3, F4])
