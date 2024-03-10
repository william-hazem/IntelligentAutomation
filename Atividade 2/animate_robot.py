import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio # save frames as gif
import os
class RobotAnimation:
    def __init__(self, x, y, orientation, trajectory, goal, title):
        temp = trajectory[:,2]
        temp = np.pi - temp
        # trajectory[:,2] = temp
        self.x = x
        self.y = y
        self.orientation = orientation
        self.trajectory = trajectory
        self.goal = np.array(goal)
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo-')
        dx = max(trajectory[:,0]) - min(trajectory[:,0])
        k = dx*0.2/2 + 1
        self.robot_vertices = np.array([[-0.1, 0.1],
                                        [0.1, 0],
                                        [-0.1, -0.1]])*k
        self.robot = plt.Polygon(self.robot_vertices, closed=True, color='red', zorder=3)
        # default= [[0, 0], [-0.5, 0.5], [0.5, 0.5]]

        # self.robot_body = plt.Polygon(default, color='red')
        self.ax.add_patch(self.robot)
        if(goal.size > 2):
            self.goal_circle = plt.Circle((goal[0][0], goal[1][0]), 0.2*k, color='green', alpha=0.5)
        else:
            self.goal_circle = plt.Circle((goal[0], goal[1]), 0.2*k, color='green', alpha=0.5)
        self.ax.add_patch(self.goal_circle)


        self.ax.set_xlim(min(trajectory[:,0])-1, max(trajectory[:,0])+1)
        self.ax.set_ylim(min(trajectory[:,1])-1, max(trajectory[:,1])+1)
        self.ax.set_aspect('equal')

        self.origin = plt.Circle((x, y), 0.2*k, color='red', alpha=0.5, zorder=0)  # Set alpha for transparency
        self.ax.add_patch(self.origin)
        self.ax.set_title(title)  # Set animation title
        self.ax.set_axis_off()
        self.frames = []

        self.animation = FuncAnimation(self.fig, self.update, frames=len(trajectory), interval=100)
        
    def update(self, frame):
        self.x, self.y, self.orientation = self.trajectory[frame]
        # self.line.set_data(self.x, self.y)
    
        # Rotation matrix
        rotation_matrix = np.array([[np.cos(self.orientation), -np.sin(self.orientation)],
                                    [np.sin(self.orientation), np.cos(self.orientation)]])

        # Rotate and translate the robot
        robot_vertices_rotated = np.dot(self.robot_vertices, rotation_matrix.T)
        robot_vertices_translated = robot_vertices_rotated + np.array([self.x, self.y])

        self.robot.set_xy(robot_vertices_translated)
        if(self.goal.size > 2):
            self.goal_circle.center = (self.goal[0][frame], self.goal[1][frame])
        else:
            self.goal_circle.center = (self.goal[0], self.goal[1])

        # save frames
        # Capture frame and store in memory
        self.fig.canvas.draw()
        buffer_frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.frames.append(buffer_frame)

        return self.line, self.robot, self.goal_circle

    def save_gif(self, filename, fps=30):
        imageio.mimsave(uri=filename, ims=self.frames, fps=fps)

# Example usage:
# def __init__():
# x = 0
# y = 0
# orientation = 0
# trajectory = np.array([[0, 0, 0], [1, 1,0], [2, 2, 0], [3, 3, 3*0], [4, 4, 0]])
# robot_anim = RobotAnimation(x, y, orientation, trajectory)
# plt.show()
