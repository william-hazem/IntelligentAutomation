# Code by William Henrique (william.marting@ee.ufcg.edu.br)
# date: 27/02/2024

# Steering control of unicycle-like robot based on 
# Lyapunov framework approach (Benbouabdallah, 2013) applied to
# tracking moving target problem

# These approach usage global frame instead a local approach bases on target frame

from numpy import cos, sin, pi, sqrt, arctan2
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from plot_result import plot_sim_result

def target_tracking_block(t, state, D, beta, alpha, vt):
    """Compute Lyapunov Control Law
        args:
        state = [e_d, e_alpha]
        pose: tuple of (robot poses array, target poses array)
    """
    global Kv, Kw
    e_d, e_alpha = state

    v = vt*cos(beta)/cos(alpha) - Kv*e_d*cos(alpha)
    v = min(abs(v), 1.2) # just depends of TT controller
    Vf = v
    w = -Kw*alpha - Vf/D*sin(alpha) + vt/D*sin(beta) 

    return v, w

def closed_loop(t, state,  target_pos, e_ref = [0, 0]):
    x_r, y_r, theta_r, v_r, x_t, y_t = state

    _, v_t = target_pos

    D, phi = compute_distance([x_r, y_r], [x_t, y_t])

    dx_t, dy_t, theta_t = target_velocity(t)
    vt = 0.5*sqrt((dx_t*cos(theta_t))**2 + (dy_t*sin(theta_t))**2)

    beta  = theta_t - phi  # target_pos[2] = theta_t
    alpha = theta_r - phi  # robot_pos[2]  = theta

    e_d = e_ref[0] - D
    e_alpha = 0 - alpha

    v, w = target_tracking_block(t, state=[e_d, e_alpha], D=D, 
                                beta=beta, alpha=alpha, vt=vt)
    print(v, w, theta_t)
    # Robotic Kinematic
    dx = v*cos(phi)
    dy = v*sin(phi)
    dtheta = w

    return [dx, dy, dtheta, v, dx_t, dy_t]


def compute_distance(robot_pos, target_pos):
    D = sqrt((robot_pos[0]-target_pos[0])**2 + (robot_pos[1] - target_pos[1])**2)
    # orientation related to target
    phi = arctan2(target_pos[1]-robot_pos[1], target_pos[0]-robot_pos[0])
    return D, phi


def circular_trajectory(t):
    if t > 10:
        return [0, 0, 0]
    return [-sin(t), cos(t), arctan2(sin(t), cos(t))]

target_velocity = lambda t: [0, 0, 0]
# target_velocity = circular_trajectory


target = [1, 1]
# robot: x, y, orientation, initial velocity
robot = [-1*sqrt(2), -1*sqrt(2), -3/4*pi, 0,
        0, 0] # target x, y
t = np.arange(0, 15, 50e-3)
Kw = 1.49
Kv = 2.07
sol = solve_ivp(closed_loop, t_span=[t[0], t[-1]], 
                t_eval=t, y0=robot, args=(target, [0, 0]))


# compute error
xy = sol.y[0:2]
theta = sol.y[2]
goal = sol.y[4:]
print(goal.shape, xy.shape)

D, phi = compute_distance(xy, goal)
alpha = theta - phi

# plt.plot(goal[0], goal[1]), plt.show()
plot_sim_result(sol.t, [D, alpha, phi],
                [xy[0], xy[1], theta],
                goal=goal, save_anim=True)
plt.show()