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
        max_thrust = env.MAX_THRUST
        # test why these are currently not working
        # max_torque_pitch_roll = env.MAX_XY_TORQUE
        # max_torque_yaw = env.MAX_Z_TORQUE
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

    def set_desired_trajectory(self, robot_idx, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        #ignore robot_idx as this is for a single robot.

        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_omega = desired_omega
        self.desired_yaw = desired_yaw

    def step_cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def skew_symmetric(self, w):
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])

    def vee_map(self, R):
        arr_out = np.zeros(3)
        arr_out[0] = -R[1, 2]
        arr_out[1] = R[0, 2]
        arr_out[2] = -R[0, 1]
        return arr_out

    def compute(self, obs, skip_low_level=False):
        x = obs_to_lin_model(obs)
        e = np.copy(x)

        #equilibrium is at the desired yaw
        R_eq = Rotation.from_euler('xyz', [0, 0, self.desired_yaw]).as_matrix()
        R = Rotation.from_euler('xyz', x[:3]).as_matrix()
        R_err = R_eq.T @ R # rotate R by the desired yaw this is the error in rotation from the equilibrium

        e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')

        #since observation and desired values are in the world frame we must rotate them
        # by the desired angle (goal frame) to compute the error between body and goal
        e[9:] = R_eq.T @ (x[9:] - self.desired_pos)
        e[6:9] = R_eq.T @ (x[6:9] - self.desired_vel)
        e[3:6] = R_eq.T @ (x[3:6] - np.array([0, 0, self.desired_omega]))

        #This is unnecessary but may help with performance so I'll leave it here for now

        #our linear model assumes that the yaw error is small, say less than pi/4 so cap the error
        # feasible_des_yaw = self.desired_yaw
        # angle_diff = (e[2] - self.desired_yaw + np.pi) % (2 * np.pi) - np.pi
        # # rotate R back by the yaw of e[2] then back to the max of angle_diff and 45
        # if abs(angle_diff) > np.pi / 8:
        #     feasible_des_yaw = e[2] + np.sign(angle_diff) * (np.pi / 8)
        #

        u = -self.K @ e
        u[0] = u[0] + self.env.M * self.env.G # offset by the equilibirum force
        action = input_to_action(self.env, u)
        return action, u
