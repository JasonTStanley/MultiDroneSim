import numpy as np
import scipy.linalg as la

import utils
from scipy.spatial.transform import Rotation
from control.base_controller import BaseController
from model.linearized import LinearizedModel
from utils.model_conversions import obs_to_lin_model, input_to_action


class LQRController(BaseController):
    def __init__(self, env, lin_model: LinearizedModel, Q=0.05 * np.eye(12), R=np.eye(4), debug=False):
        super().__init__(env)
        # try Q pos = 10, vel = 10,
        # R = .1

        # Brysons rule, essentially set Rii to be 1/(u^2_i) where u_i is the max input for the ith value)
        max_thrust = 4 * env.M * env.G
        max_torque_pitch_roll = 0.001
        max_torque_yaw = 0.001
        rflat = [1 / (max_thrust ** 2), 1 / (max_torque_pitch_roll ** 2), 1 / (max_torque_pitch_roll ** 2), 1 / (max_torque_yaw ** 2)]
        R = np.diag(rflat)
        max_vel_error = .15
        max_pos_error = .05
        max_yaw_error = np.pi / 40
        max_pitch_roll_error = np.pi / 40
        max_pitch_yaw_rate_error = .25
        max_yaw_rate_error = .25
        # stack into Q in order of state x=[r, p, y, r_dot, p_dot, y_dot, vx, vy, vz, px, py, pz]
        qflat = [1 / (max_pitch_roll_error ** 2), 1 / (max_pitch_roll_error ** 2), 1 / (max_yaw_error ** 2),
                     1 / (max_pitch_yaw_rate_error ** 2), 1 / (max_pitch_yaw_rate_error ** 2), 1 / (max_yaw_rate_error ** 2),
                     1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2),
                     1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2)]
        Q = np.diag(qflat)

        print(qflat)
        print(rflat)
        self.lin_model = lin_model
        self.Q = Q
        self.R = R
        self.debug = debug
        self.P = None
        self.K = None
        self.compute_gain_matrix()

    def compute_gain_matrix(self):
        self.P = la.solve_continuous_are(self.lin_model.A, self.lin_model.B, self.Q, self.R, e=None, s=None,
                                         balanced=True)
        # discrete time version
        # K = (R + B^T @ P B)^-1 @ B^T @ P @ A
        # self.K = la.inv(self.R + self.lin_model.B.T @ self.P @ self.lin_model.B) @ self.lin_model.B.T @ self.P @ self.lin_model.A

        # continuous time version
        # K = R^-1 @ B^T @ P
        self.K = la.solve(self.R, self.lin_model.B.T @ self.P)

    def set_desired_trajectory(self, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        # self.desired_pos = utils.to_ned @ desired_pos
        # self.desired_vel = utils.to_ned @ desired_vel
        # self.desired_yaw = -desired_yaw

        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_yaw = desired_yaw

    def step_cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def compute(self, obs):
        x = obs_to_lin_model(obs)
        e = np.copy(x)
        e[9:] = x[9:] - self.desired_pos
        e[6:9] = (x[6:9] - self.desired_vel)
        #need to take rpy -> to rot mat -> rotate by desired yaw.T -> back to rpy
        R = Rotation.from_euler('xyz', e[0:3]).as_matrix()
        Rdes = Rotation.from_euler('xyz', [0, 0, self.desired_yaw]).as_matrix()
        R = Rdes.T @ R
        e[0:3] = Rotation.from_matrix(R).as_euler('xyz')
        # e[2] = x[2] - self.desired_yaw #consider 1-cos(x2-yaw_des)
        # e[2] = np.arctan2(np.sin(e[2]), np.cos(e[2]))

        # e[2] = (e[2] + np.pi) % (2*np.pi) - np.pi # signed angle
        u = -self.K @ e
        # u[0] = -1 * u[0] # put into body frame
        # u[1:] = utils.to_ned @ u[1:]
        action = input_to_action(self.env, u)
        print(f"wz: {e[5]}, yaw: {e[2]}, action: {action}")
        if (action == np.zeros(4)).any():
            print("ACTION IS 0")
        return action
