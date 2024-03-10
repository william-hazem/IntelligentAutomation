import matplotlib.pyplot as plt
import numpy as np
from numpy import array, cos, sin, pi
from animate_robot import RobotAnimation

def plot_sim_result(t, sol, poses, path=None, save_anim = False, goal=[0, 0]):
    ""
    [x, y, phi] = poses

    states = sol
    dt = t[1]-t[0]

    e = states[0]
    alpha = states[1]
    theta = states[2]

    # alpha = cumulative_trapezoid(dalpha, t)
    # x, y, phi = states2pose(states=states)

    de = [ (e[i] - e[i-1])/dt for i in range(1, len(t))]
    de = [0] + de
    # print(len(de), len(t))
    u = -array(de)/cos(alpha)
    # x, y = states2pose(t, states, initial_pose)
    # print('(x0, y0, phi) =', x[0], y[0], phi[0])

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
    ax.legend(['$\\theta(t)$', '$\\alpha$'])

    ax = plt.subplot(3, 2, 4)
    ax.plot(t, u)
    ax.set_title("Linear velocity")
    ax.set_ylabel("$u(t) m/s$")
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

    if path is not None:
        yr = path(x)
        plt.plot(x, yr, '--')

    [plt.subplot(3,2,i).grid(True) for i in range(1, 7)]

    plt.subplots_adjust(hspace=0.45)  # Adjust the value as needed
    
    # plt.figure()
    # Plot the initial and final robot position
    pose = array([x, y, phi]).T
    robot_anim = RobotAnimation(x[0], y[0], phi[0], pose, goal, 'Robot Tracking Moving Object With Desired Distance')
    
    plt.show()
    # saving to m4 using ffmpeg writer 
    if (save_anim): robot_anim.save_gif('robot_animation.gif', fps=10)    