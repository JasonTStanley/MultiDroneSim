import numpy as np
from scipy.spatial.transform import Rotation

to_ned = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
def rpy_to_rot(rpy):
    #using conventino zyx from pg 12 of Franceso Sabatino Thesis
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                     [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R
def obs_to_lin_model(obs):
    x = np.zeros((12,))

    cur_pos = to_ned @ obs[0:3]
    cur_quat = obs[3:7]
    cur_euler_body = obs[7:10]
    cur_euler = to_ned @ cur_euler_body # essientially just multiply the y and z by a negative 1

    cur_vel = to_ned @ obs[10:13] #this should be world velocity, but probably need to convert to NED frame
    cur_ang_vel = to_ned @ obs[13:16] #this should be body angular velocity, unsure what needs to change here

    x[:3] = cur_euler
    x[3:6] = cur_ang_vel
    x[6:9] = cur_vel
    x[9:] = cur_pos
    return x

# def geo_model_to_obs(x):
#     obs = np.zeros((13,))
#     obs[:3] = x[:3]
#     obs[3:7] = Rotation.from_matrix(x[3:12].reshape((3,3))).as_quat()
#
#     obs[7:10] = x[12:15]
#     obs[10:13] = x[15:]
#     return obs

def action_to_input(env, action):
    r = env.KM / env.KF
    # convert between motor_thrusts and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
    conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                               [0.0, env.L, 0.0, -env.L],
                               [-env.L, 0.0, env.L, 0.0],
                               [-r, r, -r, r]])

    rpms = np.array(action)
    motor_thrusts = env.KF * rpms ** 2
    u = conversion_mat @ motor_thrusts
    return u

def input_to_action(env, u):
    r = env.KM / env.KF
    #if the body force is negative, set it to 0, but still control torques.
    u[0] = np.clip(u[0], 0, None)
    # convert between motor_thrusts and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
    conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                               [0.0, env.L, 0.0, -env.L],
                               [-env.L, 0.0, env.L, 0.0],
                               [-r, r, -r, r]])
    u_to_motor_thrusts = np.linalg.inv(conversion_mat)
    motor_thrusts = u_to_motor_thrusts @ u
    #ensure motor thrusts are positive
    #consider setting min thrust to something like .005*env.M*env.G
    motor_thrusts = np.clip(motor_thrusts, 0, None)
    #if motor thrusts are negative, move to minimum value
    rpms = np.sqrt(motor_thrusts / env.KF)
    action = rpms
    return action

def obs_to_geo_model(obs):
    # x = [x y z R, v_x v_y v_z w_x w_y w_z]
    # obs = [x y z q0 q1 q2 q3 vx vy vz wx wy wz]
    x = np.zeros((18,))
    x[:3] = obs[:3]
    x[3:12] = Rotation.from_quat(obs[3:7]).as_matrix().flatten()
    #obs[7:10] is rpy
    x[12:15] = obs[10:13]
    x[15:] = obs[13:16]
    return x

def geo_model_to_obs(x):
    obs = np.zeros((16,))
    obs[:3] = x[:3]
    obs[3:7] = Rotation.from_matrix(x[3:12].reshape((3,3))).as_quat()
    obs[10:13] = x[12:15]
    obs[13:16] = x[15:]
    return obs
def geo_x_dot_to_linear(geo_xdot):
    x_dot = np.zeros((12,))

    x_dot[:3] = to_ned @ geo_xdot[3:6]
    x_dot[3:6] = to_ned @ geo_xdot[9:12]
    x_dot[6:9] = to_ned @ geo_xdot[6:9]
    x_dot[9:] = to_ned @ geo_xdot[:3]
    return x_dot