import numpy as np
from numpy import sin, cos, array, tan, pi

# Define the path function
def path_function(x):
    filter = (x >= 0)
    return np.arctan(x**4) * filter

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

# Define the control law update function
def update_control_law(de, dalpha, dtheta, ds, e, theta, R):
    """Updates control law"""
    de += de + ds * np.cos(theta)
    dalpha += -ds * np.sin(theta) / e
    dtheta += -ds / R
    return de, dalpha, dtheta

# Main function for driving the robot
def drive_robot(t, prev_state, gamma, h, k, epsilon, x):
    """Implements Closed-Loop Control Law Equations"""

    e, alpha, theta = prev_state
    x = states2pose(prev_state)
    # Calculate curvature R based on current position on the path
    R = calculate_curvature(x[0])
    
    # Calculate control inputs
    dalpha = -k*alpha - (h*theta*gamma)/alpha*cos(alpha)*sin(alpha)
    dtheta = gamma*cos(alpha)*sin(alpha)
    de = -e*gamma*cos(alpha)**2
    
    # Lyapunov function
    V = gamma*e**2 + (alpha**2 + h*theta**2) 
    
    # Calculate ds
    ds = max(0, 1 - V/epsilon)
    # Update control law
    de, dalpha, dtheta = update_control_law(de, dalpha, dtheta, ds, e, theta, R)

    return array([de, dalpha, dtheta])

# Conversion function from states to pose
def states2pose(states):
    """Robot pose by given error e, and rotations"""
    e, alpha, theta = states
    x = -e*cos(theta)
    y = -e*sin(theta)
    phi = theta - alpha
    return x, y, phi

def poses2state(pose):
    x, y, phi = pose
    e = (x**2 + y**2)**.5 # considering target on origin
    theta = np.arctan(y / x)
    alpha = theta - phi
    if(alpha == 0): alpha = 1e-10
    return e, alpha, theta
# Example usage
# Define initial state and parameters

# x_initial = 0.000001

# Define the Euler integration function
def euler_integration(func, initial_state, timesteps, *args):
    """Performs Euler integration of a given function"""
    steps = 2
    
    x = np.linspace(0, 3, steps)
    y = path_function(x)
    slopes, _ = path_derivatives(x)
    
    R = lambda theta: array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    transformations = []
    T = np.eye(2, 2)
    
    targets = states2pose([x, y, 0])
    target_idx = 1
    t_idx = 0
    current_target = targets[0]
    apply_change = False
    state_offset = array([0, 0, 0])
    states = [initial_state]
    poses = [states2pose(initial_state)]
    current_state = initial_state
    dt = timesteps[1] - timesteps[0]

    for t in timesteps[1:]:
        current_state = current_state + func(t, current_state, *args) * dt
        states.append(current_state)
        current_pose = poses2state(current_state)
        

        current_error = current_state[0]
        t_idx = t_idx + 1
        if abs(current_error) < 0.003:
            print("reach at ", t)
            break
            target_idx = target_idx + 1
            xg, yg = targets[target_idx]
            x0, y0, phi0 = states2pose(current_state)
            ei, alphai, thetai = poses2state([xg-x0, yg-y0, phi0])
            print('new target', xg, yg)
            # transformations.append(R(slopes[target_idx]))
            # T = transformations[-1]*T

            # state_offset[0] = -ei
            pose = states2pose(current_state)
            current_state[0] = 2
        if target_idx > len(targets): 
            print('finished pathing')
            break
    
    return np.array(states), t

# Define simulation parameters
timesteps = np.arange(0, 100, 50e-3)  # 1000 timesteps from 0 to 10 seconds

def angle_rerange(angle):
    while abs(angle) > pi:
        angle -= pi
    return angle


gamma = 1
h = 4
k = 1
epsilon = 0.03

# Simulate the system
print("--> target 1")
x = -.5
y = path_function(x)
x_initial = 0.0001
slope = calculate_curvature(x_initial)
# phi = np.arctan(1/slope)
phi = 0
initial_state = poses2state([x, y, phi])
print('target (x, y, phi) = ', x, y, phi*180/pi)
# Assuming initial position at the origin and orientation aligned with x-axis

states_history1, t1 = euler_integration(drive_robot, initial_state, timesteps, gamma, h, k, epsilon, x_initial)

timesteps = np.linspace(t1, t1+10, 1000)  # 1000 timesteps from 0 to 10 seconds
# without change of coordsystem, phi is the same
x = 1
y = path_function(x)

print('target (x, y) = ', x, y)
x0, y0, phi = states_history1[-1,:]
x_initial = x0 # last parking point
e = ((x-x0)**2 + (y-y0)**2)**.5
print("x0, y0 =", x0, y0, phi, "e = ", e)
theta = np.arctan2(y0 - y, x0 - x)
alpha = theta - phi
initial_state = [-e, alpha, theta]
print(initial_state)
states_history2, t2 = euler_integration(drive_robot, initial_state, timesteps, gamma, h, k, epsilon, x_initial)


# Convert states to poses
poses_history = states2pose(states_history1.T)

# Extract relevant information for analysis or visualization
x_positions1 = poses_history[0]
y_positions1 = poses_history[1]
x_positions1 =  x_positions1 - x_positions1[0]
y_positions1 =  y_positions1 - y_positions1[0]

poses_history = states2pose(states_history2.T)
x_positions2 = poses_history[0]
y_positions2 = poses_history[1]
x_positions2 += - x_positions2[0] - x_positions1[-1]
y_positions2 += - y_positions2[0] - y_positions1[-1]
print(len(x_positions1), x_positions1[-5])
x_positions = np.concatenate([x_positions1, x_positions2])
y_positions = np.concatenate([y_positions1, y_positions2 ])
import matplotlib.pyplot as plt
x = np.linspace(-1, 1.5, 100)
# plt.figure()
# plt.plot(x_positions, y_positions)
# plt.plot(x, path_function(x), '--')

plt.figure()
plt.plot(x_positions1, y_positions1)
plt.plot(x, path_function(x), '--')
plt.figure()
plt.plot(x_positions2, y_positions2)
plt.plot(x, path_function(x), '--')
plt.show()