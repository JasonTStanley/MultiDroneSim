import numpy as np
import scipy.integrate
import scipy.linalg as la
from scipy.spatial.transform import Rotation

from control.lqr import lqr_controller
from control.base_controller import BaseController
from model.linearized import LinearizedModel
from utils import obs_to_lin_model, input_to_action


class DecentralizedLQR(BaseController):
    def __init__(self, env, lin_models: [LinearizedModel]):
        super().__init__(env)

        # Brysons rule, essentially set Rii to be 1/(u^2_i) where u_i is the max input for the ith value)
        max_thrust = env.MAX_THRUST
        max_torque_pitch_roll = 0.001
        max_torque_yaw = 0.001
        rflat = [1 / (max_thrust ** 2), 1 / (max_torque_pitch_roll ** 2), 1 / (max_torque_pitch_roll ** 2),
                 1 / (max_torque_yaw ** 2)]
        max_vel_error = .15
        max_pos_error = .05
        max_yaw_error = np.pi / 20
        max_pitch_roll_error = np.pi / 10
        max_pitch_yaw_rate_error = .5
        max_yaw_rate_error = .5
        # stack into Q in order of state x=[r, p, y, r_dot, p_dot, y_dot, vx, vy, vz, px, py, pz]
        qflat = [1 / (max_pitch_roll_error ** 2), 1 / (max_pitch_roll_error ** 2), 1 / (max_yaw_error ** 2),
                 1 / (max_pitch_yaw_rate_error ** 2), 1 / (max_pitch_yaw_rate_error ** 2),
                 1 / (max_yaw_rate_error ** 2),
                 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2),
                 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2)]

        self.lin_models = lin_models
        self.num_robots = len(lin_models)

        self.pred_errors = [[] for _ in range(2*self.num_robots)] # add extra set for the gt linear model predictions
        self.pred_thetas = [[] for _ in range(self.num_robots)]
        self.ind_Q = np.diag(qflat)
        self.ind_R = np.diag(rflat)
        self.Q = np.kron(np.eye(self.num_robots), self.ind_Q)  # duplicate the Q and R matrices for each robot
        self.R = np.kron(np.eye(self.num_robots), self.ind_R)
        robot_12_xy = np.index_exp[9:11, 21:23]
        robot_21_xy = np.index_exp[21:23, 9:11]
        robot_12_vxy = np.index_exp[6:8, 18:20]
        robot_21_vxy = np.index_exp[18:20, 6:8]

        # set some positive weight for the xy position of the robots
        multi_xy_err = .1
        multi_vxy_err = 0
        self.Q[robot_12_xy] = -1 / (multi_xy_err ** 2)
        self.Q[robot_21_xy] = -1 / (multi_xy_err ** 2)
        # self.Q[robot_12_vxy] = 1/(multi_vxy_err**2)
        # self.Q[robot_21_vxy] = 1/(multi_vxy_err**2)
        with np.printoptions(precision=3, suppress=True, linewidth=100000):
            print(f"Full Q: \n{self.Q}")
            print(f"Full R: \n{self.R}")

        self.Astar = np.zeros((12 * self.num_robots, 12 * self.num_robots))
        self.Bstar = np.zeros((12 * self.num_robots, 4 * self.num_robots))
        for i, agent in enumerate(lin_models):
            assert agent.Ahat.shape == (12, 12)
            assert agent.Bhat.shape == (12, 4)
            # insert the agent's approximate A and B into the correct position in the Astar and Bstar matrices
            self.Astar[i * 12:(i + 1) * 12, i * 12:(i + 1) * 12] = agent.Ahat
            self.Bstar[i * 12:(i + 1) * 12, i * 4:(i + 1) * 4] = agent.Bhat
            # here Ahat is the guess of the A matrix of the linearized model

        self.theta = np.hstack([self.Astar, self.Bstar]).T
        self.V = np.repeat((20)*np.eye(16)[:, :, None], self.num_robots, axis=2).transpose(2, 0, 1)
        self.P = np.repeat(20*np.eye(16)[:, :, None], self.num_robots, axis=2).transpose(2, 0, 1)
        # try and
        for i in range(self.num_robots):
            self.P[i][-3:, -3:] = 5_000_000 * np.eye(3)
            self.V[i][-3:, -3:] = 5_000_000 * np.eye(3)
        self.lqr_controllers = [lqr_controller.LQRController(env, lin_models[i]) for i in range(self.num_robots)]
        self.K = None
        self.desired_positions = np.zeros((self.num_robots, 3))
        self.desired_vels = np.zeros((self.num_robots, 3))
        self.desired_yaws = np.zeros(self.num_robots)
        self.desired_omegas = np.zeros(self.num_robots)
        A_keep_1 = np.index_exp[(6, 7), (1, 0)]  # refers to (6,1) and (7,0) the g values in the A matrix
        # A_keep_2 = np.index_exp[0:3, 3:6]
        A_keep_2 = np.index_exp[(0, 1, 2), (3, 4, 5)]
        # A_keep_3 = np.index_exp[9:, 6:9]
        A_keep_3 = np.index_exp[(9, 10, 11), (6, 7, 8)]
        B_keep_1 = np.index_exp[3:6, 1:]
        B_keep_2 = np.index_exp[8, 0]
        self.A_keep_2 = A_keep_2
        self.A_keep_3 = A_keep_3

        self.A_mask = np.zeros((12 * self.num_robots, 12 * self.num_robots))
        self.B_mask = np.zeros((12 * self.num_robots, 4 * self.num_robots))
        for i in range(self.num_robots):
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_1] = 1
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_2] = 1
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_3] = 1
            self.B_mask[12 * i:12 * (i + 1), 4 * i:4 * (i + 1)][B_keep_1] = 1
            self.B_mask[12 * i:12 * (i + 1), 4 * i:4 * (i + 1)][B_keep_2] = 1

    def get_thetai(self, i):
        # need to extract Ai and Bi
        Ai = self.theta[i * 12:(i + 1) * 12, 12 * i:(i + 1) * 12].T
        Bi = self.theta[(12 * self.num_robots + 4 * i):(12 * self.num_robots + 4 * (i + 1)), 12 * i:(12 * (i + 1))].T
        return np.hstack([Ai, Bi]).T

    def overwrite_theta(self, theta_new, i):
        self.theta[i * 12:(i + 1) * 12, 12 * i:(i + 1) * 12] = theta_new[:12, :]
        self.theta[(12 * self.num_robots + 4 * i):(12 * self.num_robots + 4 * (i + 1)),
        12 * i:(12 * (i + 1))] = theta_new[12:, :]

    def forward_predict(self, e, u, i, Ahat=None, Bhat=None):
        if Ahat is None and Bhat is None:
            Ahat = self.get_thetai(i)[:12, :].T
            Bhat = self.get_thetai(i)[12:, :].T

        def f(t, e_i):
            return Ahat @ e_i + Bhat @ u

        y0 = e
        sol = scipy.integrate.solve_ivp(f, [0, self.env.CTRL_TIMESTEP], y0)
        return sol.y[:, -1]

    def solve_xtp1(self, e, u, i):
        Ahat = self.get_thetai(i)[:12, :].T
        Bhat = self.get_thetai(i)[12:, :].T
        I = np.eye(12)

        xtp1 = (I + Ahat * self.env.CTRL_TIMESTEP) @ e + Bhat @ u * self.env.CTRL_TIMESTEP + (
                    1 / 2.0) * Ahat @ Bhat @ u * self.env.CTRL_TIMESTEP ** 2
        return xtp1



    def theta_update2(self, phis, xtp1s):
        for i in range(self.num_robots):
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))

            V = self.V[i]
            Vinv = np.linalg.inv(V)
            th_i = self.get_thetai(i)
            pred_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i)

            #store the prediction error
            self.pred_errors[i].append(np.linalg.norm(x_tp1.T - pred_xtp1 ))
            #this x_tp1.T - pred_xtp1 correctly shows the biggest error in z_velocity, which means the error is related to the mass estimate
            theta_new = th_i + Vinv @ phi @ (x_tp1.T - pred_xtp1)

            self.overwrite_theta(theta_new, i)
            self.project_theta()
            Ahat = self.get_thetai(i)[:12, :].T
            Bhat = self.get_thetai(i)[12:, :].T
            self.pred_thetas[i].append(np.hstack([Ahat, Bhat]))
            self.V[i] = V + phi @ phi.T
            pred_gt_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i, Ahat=self.lin_models[i].A, Bhat=self.lin_models[i].B)
            self.pred_errors[i+self.num_robots].append(np.linalg.norm(x_tp1.T-pred_gt_xtp1))


    def theta_update(self, phis, xtp1s):
        for i in range(self.num_robots):
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))

            P = self.P[i]
            L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)

            th_i = self.get_thetai(i)
            pred_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i)
            # pred_xtp1 = self.solve_xtp1(x_tp1, phi[-4:], i).flatten()

            # diff = np.linalg.norm(pred_xtp1_2 - pred_xtp1)
            # if diff > 1e-4:
            #     print("difference in integration: " + str(diff))

            #store the prediction error
            self.pred_errors[i].append(np.linalg.norm(x_tp1.T - pred_xtp1 ))
            theta_new = th_i + L @ (x_tp1.T - pred_xtp1)
            self.overwrite_theta(theta_new, i)
            Ahat = self.get_thetai(i)[:12, :].T
            Bhat = self.get_thetai(i)[12:, :].T
            self.pred_thetas[i].append(np.hstack([Ahat, Bhat]))
            self.P[i] = (np.eye(16) - L @ phi.T) @ P  # update P
            pred_gt_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i, Ahat=self.lin_models[i].A, Bhat=self.lin_models[i].B)
            self.pred_errors[i+self.num_robots].append(np.linalg.norm(x_tp1.T-pred_gt_xtp1))
        self.project_theta()
    def est_x_dot(self, x_tp1, phi):
        # TODO experiment with how this is calculated
        phi = phi.reshape((16,))
        x_tp1 = x_tp1.reshape((12,))
        x_dot = np.zeros((12,))  # estimate x_dot using what we know about the system
        # x_dot[0:3] = (phi[3:6] + x_tp1[3:6]) / 2.0  # avg of the two ang vels
        x_dot[0:3] = x_tp1[3:6]  # avg of the two ang vels
        x_dot[3:6] = (x_tp1[3:6] - phi[3:6]) / self.env.CTRL_TIMESTEP  # estimate the angular acceleration

        x_dot[6:9] = (x_tp1[6:9] - phi[6:9]) / self.env.CTRL_TIMESTEP  # estimate the linear acceleration
        # x_dot[9:] = (phi[6:9] + x_tp1[6:9]) / 2.0  # avg of the two linear vels
        x_dot[9:] = x_tp1[6:9]

        return x_dot

    def approx_theta_update(self, phis, xtp1s):
        for i in range(self.num_robots):
            # we need to estimate x_dot from observations of x_tp1 so we regress to the continuous model
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))
            x_dot = self.est_x_dot(x_tp1, phi)  # estimate x_dot

            P = self.P[i]
            L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)
            # print("L norm: " + str(np.linalg.norm(L)))
            th_i = self.get_thetai(i)
            theta_new = th_i + L @ (x_dot.T - phi.T @ th_i)
            # update the corresponding parts of theta

            # self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.overwrite_theta(theta_new, i)
            self.project_theta()
            self.P[i] = (np.eye(16) - L @ phi.T) @ P  # update P

            Ahat = self.get_thetai(i)[:12, :].T
            Bhat = self.get_thetai(i)[12:, :].T
            self.pred_thetas[i].append(np.hstack([Ahat, Bhat]))

            # pred_gt_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i, Ahat=self.lin_models[i].A,
            #                                     Bhat=self.lin_models[i].B)
            self.pred_errors[i].append(np.linalg.norm(x_dot.T - phi.T @ th_i))
            th_i_gt = np.hstack([self.lin_models[i].A, self.lin_models[i].B]).T
            self.pred_errors[i + self.num_robots].append(np.linalg.norm(x_dot.T - phi.T @ th_i_gt))
    def approx_theta_update2(self, phis, xtp1s):
        for i in range(self.num_robots):
            # we need to estimate x_dot from observations of x_tp1 so we regress to the continuous model
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))
            x_dot = self.est_x_dot(x_tp1, phi)  # estimate x_dot
            V = self.V[i]
            # P = self.P[i]
            Vinv = np.linalg.inv(V)
            # L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)
            # print("L norm: " + str(np.linalg.norm(L)))
            th_i = self.get_thetai(i)
            theta_new = th_i + Vinv @ phi @ (x_dot.T - phi.T @ th_i)

            # update the corresponding parts of theta

            # self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.overwrite_theta(theta_new, i)
            self.project_theta()
            # self.P[i] = (np.eye(16) - L @ phi.T) @ P  # update P
            self.V[i] = V + phi @ phi.T
            Ahat = self.get_thetai(i)[:12, :].T
            Bhat = self.get_thetai(i)[12:, :].T
            self.pred_thetas[i].append(np.hstack([Ahat, Bhat]))

            # pred_gt_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i, Ahat=self.lin_models[i].A,
            #                                     Bhat=self.lin_models[i].B)
            self.pred_errors[i].append(np.linalg.norm(x_dot.T - phi.T @ th_i))
            th_i_gt = np.hstack([self.lin_models[i].A, self.lin_models[i].B]).T
            self.pred_errors[i + self.num_robots].append(np.linalg.norm(x_dot.T - phi.T @ th_i_gt))


    def project_theta(self):
        # set the known zeros to zero
        Ahat = self.theta[:-(4*self.num_robots), :].T
        Bhat = self.theta[-(4*self.num_robots):, :].T
        Ahat = Ahat * self.A_mask
        Bhat = Bhat * self.B_mask
        for i in range(self.num_robots):
            Ahat[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][self.A_keep_2] = 1
            Ahat[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][self.A_keep_3] = 1

        self.theta = np.hstack([Ahat, Bhat]).T

    def sigma1(self):
        # draw a random input force and thrust, ensure that the input is within the bounds
        # thrust must be between 0 and 4*env.M*env.G, torques must be between -0.001 and 0.001
        thrust = np.random.uniform(.8 * self.env.M * self.env.G, 1.5 * self.env.M * self.env.G)
        torques = np.random.uniform(-0.00001, 0.00001, 3)
        return np.hstack([thrust, torques])

    def sigma_explore(self):
        # do gausian noise around hover with some small covariance for the input
        thrust_cov = .15 * self.env.M * self.env.G
        torque_xy_cov = 0.005 * self.env.MAX_XY_TORQUE
        torque_z_cov = 0.005 * self.env.MAX_Z_TORQUE

        # thrust_cov = .5 * self.env.M * self.env.G
        # torque_xy_cov = 0.04 * self.env.MAX_XY_TORQUE
        # torque_z_cov = 0.04 * self.env.MAX_Z_TORQUE


        # torque_xy_cov = 0.00000001
        # torque_z_cov = 0.00000001
        # thrust_cov = .5 * self.env.M * self.env.G
        # torque_xy_cov = 0.01 * self.env.MAX_XY_TORQUE
        # torque_z_cov = 0.01 * self.env.MAX_Z_TORQUE
        thrust = np.random.normal(self.env.M * self.env.G, thrust_cov)
        torques = np.random.normal(0, [torque_xy_cov, torque_xy_cov, torque_z_cov])
        # may need to bound system. ie if we have moved too far in one direction, only allow exploration in the other direction
        return np.hstack([thrust, torques])

    def noisy_control(self, obs):
        action, u = self.compute(obs)
        thrust = np.random.uniform(.01 * self.env.M * self.env.G,
                                   0.01 * self.env.M * self.env.G)  # add noise to the thrust
        torques = np.random.uniform(-0.000001, 0.000001, 3)  # add noise to the torques
        ctrl_noise = np.hstack([thrust, torques])

        u_noisy = u + ctrl_noise
        action_noisy = input_to_action(self.env, u_noisy)
        return action_noisy, u_noisy

    def LQR(self, obs, robot_idx, pos=np.array([0, 0, 0]), yaw=0, vel=np.array([0, 0, 0]), omega=0):
        self.lqr_controllers[robot_idx].set_desired_trajectory(robot_idx, desired_pos=pos, desired_vel=vel,
                                                               desired_acc=[0, 0, 0], desired_yaw=yaw,
                                                               desired_omega=omega)
        action = self.lqr_controllers[robot_idx].compute(obs)
        return action

    def error_state(self, x, x_des):
        e = np.copy(x)

        R_eq = Rotation.from_euler('xyz', [0, 0, x_des[2]]).as_matrix()
        R = Rotation.from_euler('xyz', x[:3]).as_matrix()
        R_err = R_eq.T @ R
        e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')
        e[9:] = R_eq.T @ (x[9:] - x_des[9:])
        e[6:9] = R_eq.T @ (x[6:9] - x_des[6:9])
        e[3:6] = R_eq.T @ (x[3:6] - x_des[3:6])
        return e

    def compute_controller(self, force_diagonal=False):
        # extract the A and B matrices from the theta matrix and compute K
        if force_diagonal:
            self.K = np.zeros((4 * self.num_robots, 12 * self.num_robots))
            for i in range(self.num_robots):
                A = self.theta[:12 * self.num_robots, :].T[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)]
                B = self.theta[12 * self.num_robots:, :].T[12 * i:12 * (i + 1), 4 * i:4 * (i + 1)]
                # A = self.Astar[12*i:12*(i+1), 12*i:12*(i+1)] testing with baseline A,B instead of learned
                # B = self.Bstar[12*i:12*(i+1), 4*i:4*(i+1)]
                P = la.solve_continuous_are(A, B, self.ind_Q, self.ind_R, e=None, s=None, balanced=True)
                K_i = la.solve(self.R, B.T @ P)
                self.K[4 * i:4 * (i + 1), 12 * i:12 * (i + 1)] = K_i
        else:

            A = self.theta[:12 * self.num_robots, :].T  # ensure that this extracs A and B properly
            B = self.theta[12 * self.num_robots:, :].T
            P = la.solve_continuous_are(A, B, self.Q, self.R, e=None, s=None, balanced=True)
            self.K = la.solve(self.R, B.T @ P)

    def set_desired_trajectory(self, robot_idx, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        self.desired_positions[robot_idx] = desired_pos
        self.desired_vels[robot_idx] = desired_vel
        self.desired_yaws[robot_idx] = desired_yaw
        self.desired_omegas[robot_idx] = desired_omega


    def compute(self, obs, skip_low_level=False):
        es = [self.error_state(obs_to_lin_model(obs[i]),
                               np.hstack([np.array([0, 0, self.desired_yaws[i]]),
                                          np.array([0, 0, self.desired_omegas[i]]), self.desired_vels[i],
                                          self.desired_positions[i]]))
              for i in range(self.num_robots)]
        # select out the correct A and B matrices for the robot
        us = np.array([-self.K[:, (12 * i):12 * (i + 1)] @ es[i] for i in range(self.num_robots)])
        # if us[0][-4:][2] < 0:
        #     print("effort to evade (ty): " + str(us[0][-4:][2]))
        u = np.sum(us, axis=0)

        u_robot = np.array([u[(4 * i):(4 * (i + 1))] for i in range(self.num_robots)])
        # offset by the equilibrium thrust
        u_robot[:, 0] += self.env.M * self.env.G
        action = np.array([input_to_action(self.env, u) for u in u_robot])
        return action, u

    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u
