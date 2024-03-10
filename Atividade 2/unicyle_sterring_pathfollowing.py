# Code by William Henrique (william.marting@ee.ufcg.edu.br)
# date: 22/02/2024

# Steering control of unicycle-like robot based on 
# Lyapunov framework approach (Aicardi, 1994), (Aicardi, 1995)

from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid

import numpy as np
from numpy import cos, sin, array, pi, arctan
from matplotlib import pyplot as plt
from plot_result import plot_sim_result

def drive_steering(t, prev_state, gamma, h, k):
    "Implements Closed-Loop Control Law Equations"

    e, alpha, theta = prev_state
    # system states
    dalpha = -k*alpha - (h*theta*gamma)/alpha*cos(alpha)*sin(alpha)
    dtheta = gamma*cos(alpha)*sin(alpha)
    # system output
    de = -e*gamma*cos(alpha)**2
    # u = gamma*cos(alpha)*e
    u = dtheta*e/sin(alpha)
    return array([de, dalpha, dtheta])

def linear_drive_steering(t, prev_state, gamma, h, k):
    "Implements Linearization of Robot Steering Control Law"
    e, alpha, theta = prev_state
    de = -gamma*e
    dalpha = -k*alpha - gamma*h*theta
    dtheta = gamma*alpha
    u = gamma*e
    return array([de, dalpha, dtheta])

def coord2states(pose):
    x, y, phi = pose
    e = (x**2 + y**2)**.5 # considering target on origin
    theta = arctan(y / x)
    alpha = theta - phi
    if(alpha == 0): alpha = 1e-10
    return [e, alpha, theta]

def states2pose(states):
    """Robot pose by given error e, and rotations"""
    e, alpha, theta = states
    x = -e*cos(theta)
    y = -e*sin(theta)
    phi = theta-alpha
    return x, y, phi

def path_function(x):
    # Define the path function
    return np.tan(x**4) 

# Define the first and second derivatives of the path function
def path_derivatives(x):
    dy_dx = 4 * x**3 / (1 + x**8)  # First derivative of y = tan(x^4)
    d2y_dx2 = (4*x**2*(3-5*x**8)) / \
          ((x**8 + 1)**2)  # Second derivative of y = tan(x^4)
    return dy_dx, d2y_dx2

# Define the function to calculate curvature R
def calculate_curvature(x):
    dy_dx, d2y_dx2 = path_derivatives(x)
    curvature = ((1 + dy_dx**2)**(3/2)) / abs(d2y_dx2)
    return curvature   

def path_following(t, prev_state, gamma, h, k):
    global __dx0
    e, alpha, theta = prev_state
    # system states
    dalpha = -k*alpha - (h*theta*gamma)/alpha*cos(alpha)*sin(alpha)
    dtheta = gamma*cos(alpha)*sin(alpha)
    # system output
    de = -e*gamma*cos(alpha)**2

    V =  gamma*e**2 + (alpha**2 + h*theta**2)
    ds = max(0, 1-V/epsilon)
    ds_array.append(ds)
    # Aprox. Curvature for y = atan(x^4)
    R = None # |y''|/(1 + y'^2)^(3/2)
    dx = -e*gamma*cos(theta)**2
    x = (dx - __dx0) * dt + x0
    dy = 4*x**3/(1+x**8)
    ddy = 4*x**2*(3-5*x**8)/(x**8 + 1)**2
    __dx0 += dx

    k = (ddy) / (1 + dy**2)**(3/2)
    R = 1/k
    print(x, R, np.arctan(dy)*180/pi)

    # updates control law
    
    de     += de + ds*cos(theta)
    dalpha += -ds*sin(theta)/e
    dtheta += -ds/R
    return [de, dalpha, dtheta]
__dx0 = 0
step_e = 0
ds_array = []
initial_pose = [-1, 40, 3*pi/4]
initial_state = [10, 0.0075, 0.01]

initial_state = [2*np.sqrt(2), -pi, -pi/4]
x0 = -1
# print('pose', [1, path_function(1), 1])
initial_state = coord2states([x0, path_function(x0), -pi/2])

# Closed Loop parameters
epsilon = 0.01
gamma = .1
h = 50
k = 1

dt = 50e-3
t = np.arange(0, 20, dt)

sol = solve_ivp(fun = path_following, t_span=[t[0], t[-1]], y0=initial_state, t_eval=t,
                method='RK45',  # Runge-Kutta
                args=[gamma, h, k]
)
print('ds', len(ds_array), 't', len(t))
# sol = odeint(drive_steering, initial_state, t)

# print(sol.y[2])
state_ = array(sol.y)
print(state_.shape)
(x, y, phi) = states2pose(state_)
print(x.shape)
# x = x - x[0]
# y = y - y[0]
# phi = phi - phi[0]
print(ds_array)
print("first pose", x[0], y[0], phi[0])
print("last pose",  x[-1], y[-1], phi[-1])
plot_sim_result(sol.t, sol.y, [x, y, phi], path_function)
