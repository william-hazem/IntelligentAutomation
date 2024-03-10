# Code by William Henrique (william.marting@ee.ufcg.edu.br)
# date: 22/02/2024

# Steering control of unicycle-like robot based on 
# Lyapunov framework approach (Aicardi, 1994), (Aicardi, 1995)

from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid

import numpy as np
from numpy import cos, sin, array, pi, arctan
from matplotlib import pyplot as plt
from animate_robot import RobotAnimation

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
    return [e, alpha, theta]

def states2pose(states):
    e, alpha, theta = states
    x = -e*cos(theta)
    y = -e*sin(theta)
    phi = theta-alpha
    return x, y, phi

def check_stability(gamma, h, k):
    # implements a conditional check about solution stability
    # and return system poles
    
    assert(h > 1, "h must be greater than 1")
    assert(k > 2*gamma, "k must be greater than 2*gamma = " + str(2*gamma))
    assert(k < gamma*(h+1), "k must be lesser than gamma(h+1) = " + str(gamma*(h+1)))
    wd = np.sqrt(k**2 - 4*h*gamma**2)
    lambda1 = -k/2 + wd/2
    lambda2 = -k/2 - wd/2
    return lambda1, lambda2

def plot_sim_result(sol):
    t = sol.t
    states = sol.y
    dt = t[1]-t[0]

    e = states[0]
    alpha = states[1]
    theta = states[2]

    # alpha = cumulative_trapezoid(dalpha, t)
    x, y, phi = states2pose(states=states)

    de = [ (e[i] - e[i-1])/dt for i in range(1, len(t))]
    de = [0] + de
    print(len(de), len(t))
    u = -array(de)/cos(alpha)
    # x, y = states2pose(t, states, initial_pose)
    print('(x0, y0, phi) =', x[0], y[0], phi[0])

    plt.figure()
    ax = plt.subplot(3, 2, 2)
    ax.plot(t, e)
    ax.set_title("Robot steering error")
    ax.set_ylabel("e(t)")

    ax = plt.subplot(3, 2, 6)
    ax.plot(t, theta*180/pi, t, alpha*180/pi)
    ax.set_title("Trajectory orientation")
    ax.set_ylabel("$\\theta(t), \\alpha$ deg")
    ax.set_xlabel("$t (s) $")
    ax.legend(['$\\theta(t)$', '\\alpha'])

    ax = plt.subplot(3, 2, 4)
    ax.plot(u)
    ax.set_title("Linear velocity")
    ax.set_ylabel("$\\alpha(t) deg$")
    ax.set_xlabel("$t (s) $")

    ax = plt.subplot(3, 2, 1)
    ax.set_title("Robot x-pose over time")
    ax.plot(t, x)
    ax.set_ylabel("$x(t)$")
   
    ax = plt.subplot(3, 2, 3)
    # ax.plot(t, sol.y[2])
    ax.plot(t, y)
    ax.set_title("Robot y-pose over time")
    ax.set_ylabel("$y(t)$")
    ax.set_xlabel("t (s)")

    ax = plt.subplot(3, 2, 5)
    plt.plot(x, y)
    plt.title('Robot Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')

    [plt.subplot(3,2,i).grid(True) for i in range(1, 7)]
    plt.grid(True)
    plt.subplots_adjust(hspace=0.45)  # Adjust the value as needed
    
    # plt.figure()
    # Plot the initial and final robot position
    pose = array([x, y, phi]).T
    print(pose.shape)
    robot_anim = RobotAnimation(x[0], y[0], phi[0], pose, array([0, 0]), title="")
    
    plt.show()
    # saving to m4 using ffmpeg writer 
    robot_anim.save_gif('robot_animation.gif', fps=10)
    

initial_pose = [-1, 40, 3*pi/4]
initial_state = [10, 0.0075, 0.01]

initial_state = coord2states([-2, 0, 0])
initial_state = [1*np.sqrt(2), pi, -pi/4]

# Closed Loop parameters
gamma = 1
h = 50
k = 50

l1, l2 = check_stability(gamma, h, k)
sigma = max(l1, l2)          
print('Dominant pole =', sigma)

dt = 50e-3
t = np.arange(0, 5, dt)

sol = solve_ivp(fun = drive_steering, t_span=[t[0], t[-1]], y0=initial_state, t_eval=t,
                method='RK45',  # Runge-Kutta
                args=[gamma, h, k]
)

# sol = odeint(drive_steering, initial_state, t)

# print(sol.y[2])
plot_sim_result(sol)