import numpy as np
import scipy.linalg as la
from scipy.spatial.transform import Rotation

import utils
from control.low_level.yank_omega_ctrl import YankOmegaController
from control.base_controller import BaseController
from model.linear_yank_omega import LinearizedYankOmegaModel
from utils.model_conversions import obs_to_lin_model


class LQRYankOmegaController(BaseController):
    def __init__(self, env, lin_model: LinearizedYankOmegaModel, yo_controller: YankOmegaController, debug=False, use_noisy_model=False):
        super().__init__(env)
        self.yo_controller = yo_controller
        # Brysons rule, essentially set Rii to be 1/(u^2_i) where u_i is the max input for the ith value)

        max_yank = (env.MAX_THRUST / env.CTRL_TIMESTEP ) / 200 # guess? don't have an intuition for yank yet

        max_pitch_roll_rate_error = 0.1
        max_yaw_rate_error = 0.1
        rflat = [1 / (max_yank ** 2), 1 / (max_pitch_roll_rate_error ** 2), 1 / (max_pitch_roll_rate_error ** 2),
                 1 / (max_yaw_rate_error ** 2)]
        R = np.diag(rflat)
        max_vel_error = .15
        max_pos_error = .05
        max_yaw_error = np.pi / 40
        max_pitch_roll_error = np.pi / 20
        max_thrust = env.MAX_THRUST - env.M* env.G #max thrust in equilibrium input
        # stack into Q in order of state x=[r, p, y, T, vx, vy, vz, px, py, pz] (T is thrust)
        qflat = [1 / (max_pitch_roll_error ** 2), 1 / (max_pitch_roll_error ** 2), 1 / (max_yaw_error ** 2),
                 1/(max_thrust**2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2),
                 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2)]
        Q = np.diag(qflat)

        self.lin_model = lin_model
        self.Q = Q
        self.R = R
        self.debug = debug
        self.P = None
        self.K = None
        self.use_noisy_model = use_noisy_model
        if use_noisy_model:
            self.A = self.lin_model.Ahat
            self.B = self.lin_model.Bhat
        else:
            self.A = self.lin_model.A
            self.B = self.lin_model.B
        self.compute_gain_matrix()


    def compute_gain_matrix(self):
        self.P = la.solve_continuous_are(self.A, self.B, self.Q, self.R, e=None, s=None,
                                         balanced=True)
        # continuous time version
        # K = R^-1 @ B^T @ P
        self.K = la.solve(self.R, self.B.T @ self.P)

    def set_desired_trajectory(self, robot_idx, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        # ignore robot_idx as this is for a single robot.

        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
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

    def compute_low_level(self, u, obs, idx=0):
        cur_quat = obs[3:7]
        cur_ang_vel_w = obs[13:16]
        R = Rotation.from_quat(cur_quat).as_matrix()
        # convert omega from world to body frame
        cur_ang_vel_b = R.T @ cur_ang_vel_w

        action = self.yo_controller.computeControlFromInput(u=u, control_timestep=self.env.CTRL_TIMESTEP,
                                                            cur_ang_vel=cur_ang_vel_b,
                                                            cur_thrust=utils.calc_z_thrust(self.env, obs))

        return action

    def compute(self, obs, skip_low_level=False):
        x = obs_to_lin_model(obs, dim=10, env=self.env)
        # drop angular velocity from the state

        e = np.copy(x)

        # equilibrium is at the desired yaw
        R_eq = Rotation.from_euler('xyz', [0, 0, self.desired_yaw]).as_matrix()
        R = Rotation.from_euler('xyz', x[:3]).as_matrix()
        R_err = R_eq.T @ R  # rotate R by the desired yaw this is the error in rotation from the equilibrium

        e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')
        e[3] = x[3] - self.env.M * self.env.G
        # since observation and desired values are in the world frame we must rotate them
        # by the desired angle (goal frame) to compute the error between body and goal
        e[7:] = R_eq.T @ (x[7:] - self.desired_pos)
        e[4:7] = R_eq.T @ (x[4:7] - self.desired_vel)

        u = -self.K @ e
        if skip_low_level:
            return None, u
        action = self.compute_low_level(u,obs)
        # action = input_to_action(self.env, u)
        return action, u

    def cap_u(self, u):
        #potentially cap the max Yank
        pass

    def get_yank(self):
        return