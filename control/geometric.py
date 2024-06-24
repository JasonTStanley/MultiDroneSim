import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation
from utils import obs_to_geo_model, input_to_action
from control.base_controller import BaseController
class GeometricControl(BaseController):
    def __init__(self, env):
        super().__init__(env)
        mass = env.M
        inertia = env.J
        self.m = mass
        self.J = inertia

        self.Kp = np.array([2.25, 2.25, 2.25])
        self.Kv = np.array([3.5, 3.5, 3.5])
        self.KR = np.array([125, 125, 125])
        self.Kw = np.array([10, 10, 10])
        self.e3 = np.array([0., 0., 1.])

        self.g = 9.81
        self.max_force = 4 * self.m * self.g
        self.max_torque = 0.1
        self.max_tilt_angle = 40 * np.pi / 180

        self.desired_position = None
        self.desired_rotation = None
        self.desired_velocity = None
        self.desired_acceleration = None

    def hat_map(self, w):
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
        return w_hat

    def vee_map(self, R):
        '''
        Performs the vee mapping from a rotation matrix to a vector
        '''
        arr_out = np.zeros(3)
        arr_out[0] = -R[1, 2]
        arr_out[1] = R[0, 2]
        arr_out[2] = -R[0, 1]
        return arr_out

    def set_desired_trajectory(self, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        '''
        desired_vel is represented in the world-frame
        desired_omega is represented in the body-frame
        '''
        self.desired_position = desired_pos
        self.desired_velocity = desired_vel
        self.desired_acceleration = desired_acc
        self.desired_yaw = desired_yaw
        self.desired_omega = desired_omega

    def compute(self, obs):
        '''
        NOTES:
        - Velocities are represented in the body-frame
        '''
        current_state = obs_to_geo_model(obs)
        p, R, v, w = current_state[:3], current_state[3:12], current_state[12:15], current_state[15:]
        p_des, v_des, a_des = self.desired_position, self.desired_velocity, self.desired_acceleration

        R = R.reshape(3, 3)
        RT = R.T
        v = RT @ v
        w_hat = self.hat_map(w)

        f_des_body = RT @ (self.m * self.g * self.e3) - self.m * RT @ np.diag(self.Kp) @ (p - p_des) - self.m * np.diag(
            self.Kv) @ (v - RT @ v_des) + self.m * (RT @ a_des - w_hat @ RT @ v_des)
        f_des_world = R @ f_des_body

        # Limit tilt angle
        tilt_angle = np.arccos(f_des_world[2] / np.linalg.norm(f_des_world))
        if tilt_angle > self.max_tilt_angle:
            xy_mag = np.linalg.norm(f_des_world[:2])
            xy_mag_max = f_des_world[2] * np.tan(self.max_tilt_angle)
            scale_acc = xy_mag_max / xy_mag
            f_des_world[:2] = f_des_world[:2] * scale_acc
        f_des_body = np.matmul(RT, f_des_world)

        # Desired rotation
        b1c = np.array([np.cos(self.desired_yaw), np.sin(self.desired_yaw), 0])
        b3d = f_des_world / np.linalg.norm(f_des_world)
        b2d = np.cross(b3d, b1c) / np.linalg.norm(np.cross(b3d, b1c))
        b1d = np.cross(b2d, b3d) / np.linalg.norm(np.cross(b2d, b3d))
        R_des = np.vstack((b1d, b2d, b3d)).T

        # Desired angular velocity
        b1c_dot = np.array([-np.sin(self.desired_yaw) * self.desired_omega, np.cos(self.desired_yaw) * self.desired_omega, 0])
        f_des_dot_world = self.m * R @ np.diag(self.Kp) @ (v - RT @ v_des) / np.linalg.norm(f_des_world)
        b3d_dot = np.cross(np.cross(b3d, f_des_dot_world), b3d)
        b2d_dot = np.cross(
            np.cross(b2d, (np.cross(b1c_dot, b3d) + np.cross(b1c, b3d_dot)) / np.linalg.norm(np.cross(b1c, b3d))), b2d)
        b1d_dot = np.cross(b3d_dot, b2d) + np.cross(b3d, b2d_dot)
        R_dot_des = np.vstack((b1d_dot, b2d_dot, b3d_dot)).T
        w_des_hat = np.matmul(R_des.transpose(0, 1), R_dot_des)
        w_des = np.array([w_des_hat[2, 1], w_des_hat[0, 2], w_des_hat[1, 0]])

        e_R = 0.5 * np.diag(self.KR) @ self.vee_map(R_des.T @ R - R.T @ R_des)
        torque = self.J @ (-e_R - np.diag(self.Kw) @ (w - np.matmul(RT, np.matmul(R_des, w_des)))) - np.cross(w,
                                                                                                              self.J @ w)

        # Control
        u = np.hstack((max(0., f_des_body[2]), torque))
        return input_to_action(self.env, u)
