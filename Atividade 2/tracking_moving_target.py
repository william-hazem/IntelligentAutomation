# Code by William Henrique (william.marting@ee.ufcg.edu.br)
# date: 27/02/2024

# Steering control of unicycle-like robot based on 
# Lyapunov framework approach (Benbouabdallah, 2013)

# These approach usage global frame instead a local approach bases on target frame

from numpy import cos, sin, pi, sqrt, arctan2
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_result import plot_sim_result

def target_tracking_block(t, state, D, beta, alpha):
    """Compute Lyapunov Control Law
        args:
        state = [e_d, e_alpha]
        pose: tuple of (robot poses array, target poses array)
    """
    global Kv, Kw
    e_d, e_alpha = state

    vt = target_velocity(t) 

    v = vt*cos(beta)/sin(alpha) - Kv*e_d*cos(alpha)
    Vf = v # just depends of TT controller
    w = -Kw*alpha - Vf/D*sin(alpha) + vt/D*sin(beta) 

    return v, w

def closed_loop(t, state,  target_pos, e_ref = [0, 0]):
    x_r, y_r, theta_r, v_r = state

    x_t, y_t, theta_t, v_t = target_pos

    D, phi = compute_distance(state, target_pos)

    beta  = theta_t - phi  # target_pos[2] = theta_t
    alpha = theta_r - phi  # robot_pos[2]  = theta

    e_d = e_ref[0] - D
    e_alpha = 0 - alpha

    v, w = target_tracking_block(t, state=[e_d, e_alpha], D=D, beta=beta, alpha=alpha)
    
    # Robotic Kinematic
    dx = v*cos(phi)
    dy = v*cos(phi)
    dtheta = w

    return [dx, dy, dtheta, v]


def compute_distance(robot_pos, target_pos):
    D = sqrt((robot_pos[0]-target_pos[0])**2 + (robot_pos[1] - target_pos[1])**2)
    # orientation related to target
    phi = arctan2(target_pos[1]-robot_pos[1], target_pos[0]-robot_pos[0])
    return D, phi


target_velocity = lambda t: 0

# target: x, y, orientation, initial velocity
target = [0, 0, 0, 0]
# robot: x, y, orientation, initial velocity
robot = [-sqrt(2), -sqrt(2), 3/4*pi, 0]

t = np.arange(0, 10, 50e-3)
Kw = 1
Kv = 1
sol = solve_ivp(closed_loop, t_span=[t[0], t[-1]], 
                t_eval=t, y0=robot, args=(target, [0, 0]))


# compute error
xy = sol.y[0:2]
theta = sol.y[2]
print(xy.shape)

D, phi = compute_distance(xy, target[0:2])
alpha = theta - phi
# plt.plot(sol.t, sol.y[0])
plot_sim_result(t, [D, alpha, alpha],
                [xy[0], xy[1], phi],
                goal=target[:2])
plt.show()