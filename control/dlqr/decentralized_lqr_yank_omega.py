import numpy as np
import scipy.integrate
import scipy.linalg as la
from scipy.spatial.transform import Rotation

from control.lqr import lqr_omega_controller
from control.base_controller import BaseController
from model.linearized import LinearizedModel
from utils import obs_to_lin_model


class DecentralizedLQRYankOmega(BaseController):
    def __init__(self, env, lin_models: [LinearizedModel], debug=False):
        from control import ThrustOmegaController as TOC
        super().__init__(env)
        # set dimensions of model
        # A is mxm B is mxn, m=9, n=4
        self.m = 10
        self.n = 4
        self.mn = self.m + self.n
        # Brysons rule, essentially set Rii to be 1/(u^2_i) where u_i is the max input for the ith value)

        max_yank = (env.MAX_THRUST / env.CTRL_TIMESTEP) / 2  # guess? don't have an intuition for yank yet

        max_pitch_roll_rate_error = 0.1
        max_yaw_rate_error = 0.1
        rflat = [1 / (max_yank ** 2), 1 / (max_pitch_roll_rate_error ** 2), 1 / (max_pitch_roll_rate_error ** 2),
                 1 / (max_yaw_rate_error ** 2)]
        R = np.diag(rflat)
        max_vel_error = .15
        max_pos_error = .05
        max_yaw_error = np.pi / 40
        max_pitch_roll_error = np.pi / 20
        max_thrust = env.MAX_THRUST - env.M * env.G  # max thrust in equilibrium input
        # stack into Q in order of state x=[r, p, y, T, vx, vy, vz, px, py, pz] (T is thrust)
        qflat = [1 / (max_pitch_roll_error ** 2), 1 / (max_pitch_roll_error ** 2), 1 / (max_yaw_error ** 2),
                 1 / (max_thrust ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2),
                 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2)]

        self.lin_models = lin_models
        self.num_robots = len(lin_models)
        self.ind_Q = np.diag(qflat)
        self.ind_R = np.diag(rflat)
        self.Q = np.kron(np.eye(self.num_robots), self.ind_Q)  # duplicate the Q and R matrices for each robot
        self.R = np.kron(np.eye(self.num_robots), self.ind_R)
        if debug:
            with np.printoptions(precision=3, suppress=True, linewidth=100000):
                print(f"Full Q: \n{self.Q}")
                print(f"Full R: \n{self.R}")

        self.Astar = np.zeros((self.m * self.num_robots, self.m * self.num_robots))
        self.Bstar = np.zeros((self.m * self.num_robots, self.n * self.num_robots))
        for i, agent in enumerate(lin_models):
            assert agent.Ahat.shape == (self.m, self.m)
            assert agent.Bhat.shape == (self.m, self.n)
            # insert the agent's approximate A and B into the correct position in the Astar and Bstar matrices
            self.Astar[i * self.m:(i + 1) * self.m, i * self.m:(i + 1) * self.m] = agent.Ahat
            self.Bstar[i * self.m:(i + 1) * self.m, i * self.n:(i + 1) * self.n] = agent.Bhat
            # here Ahat is the guess of the A matrix of the linearized model

        self.theta = np.hstack([self.Astar, self.Bstar]).T

        self.V = np.eye(self.mn * self.num_robots)
        self.P = np.repeat(np.eye(self.mn)[:, :, None], self.num_robots, axis=2).transpose(2, 0, 1)
        self.lqr_controllers = [lqr_omega_controller.LQROmegaController(env, lin_models[i], TOC(env))
                                for i in range(self.num_robots)]
        self.K = None
        self.desired_positions = np.zeros((self.num_robots, 3))
        self.desired_vels = np.zeros((self.num_robots, 3))
        self.desired_yaws = np.zeros(self.num_robots)
        self.desired_omegas = np.zeros(self.num_robots)
        self.low_level_controllers = [ThrustOmegaController(env) for _ in range(self.num_robots)]

    def get_thetai(self, i):
        # need to extract Ai and Bi
        m = self.m
        n = self.n
        Ai = self.theta[i * m:(i + 1) * m, m * i:(i + 1) * m].T
        Bi = self.theta[(m * self.num_robots + n * i):(m * self.num_robots + n * (i + 1)), m * i:(m * (i + 1))].T
        return np.hstack([Ai, Bi]).T

    def overwrite_theta(self, theta_new, i):
        m = self.m
        n = self.n
        self.theta[i * m:(i + 1) * m, m * i:(i + 1) * m] = theta_new[:m, :]
        self.theta[(m * self.num_robots + n * i):(m * self.num_robots + n * (i + 1)), m * i:(m * (i + 1))] = theta_new[
                                                                                                             m:, :]

    def forward_predict(self, e, u, i):
        m = self.m
        Ahat = self.get_thetai(i)[:m, :].T
        Bhat = self.get_thetai(i)[m:, :].T

        def f(t, e):
            return Ahat @ e + Bhat @ u

        y0 = e
        sol = scipy.integrate.solve_ivp(f, [0, self.env.CTRL_TIMESTEP], y0)
        return sol.y[:, -1]

    def solve_xtp1(self, e, u, i):
        m = self.m
        n = self.n

        Ahat = self.get_thetai(i)[:m, :].T
        Bhat = self.get_thetai(i)[m:, :].T
        I = np.eye(m)
        xtp1 = (I + Ahat * self.env.CTRL_TIMESTEP) @ e + Bhat @ u * self.env.CTRL_TIMESTEP + (
                1 / 2.0) * Ahat @ Bhat @ u * self.env.CTRL_TIMESTEP ** 2
        return xtp1

    def theta_update(self, phis, xtp1s):
        for i in range(self.num_robots):
            x_tp1 = xtp1s[i].reshape((self.m, 1))
            phi = phis[i].reshape((self.mn, 1))

            P = self.P[i]
            L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)

            th_i = self.get_thetai(i)
            pred_xtp1 = self.forward_predict(x_tp1.flatten(), phi[-self.n:].flatten(), i)
            # pred_xtp1 = self.solve_xtp1(x_tp1, phi[-4:], i).flatten()

            theta_new = th_i + L @ (x_tp1.T - pred_xtp1)
            self.overwrite_theta(theta_new, i)
            self.P[i] = (np.eye(self.mn) - L @ phi.T) @ P  # update P

    def sigma1(self):
        # draw a random input force and thrust, ensure that the input is within the bounds
        thrust = np.random.uniform(.7 * self.env.M * self.env.G, 1.5 * self.env.M * self.env.G)
        ang_vs = np.random.uniform(-0.00001, 0.00001, 3)
        return np.hstack([thrust, ang_vs])

    def sigma_explore(self):
        # do gausian noise around hover with some small covariance for the input
        thrust_cov = .005 * self.env.M * self.env.G
        angv_xy_cov = 0.000000005
        angv_z_cov = 0.000000005
        thrust = np.random.normal(self.env.M * self.env.G, thrust_cov)
        angv = np.random.normal(0, [angv_xy_cov, angv_xy_cov, angv_z_cov])
        # may need to bound system. ie if we have moved too far in one direction, only allow exploration in the other direction
        return np.hstack([thrust, angv])

    def noisy_control(self, obs, robot_idx, pos=np.array([0, 0, 0]), yaw=0, vel=np.array([0, 0, 0]), omega=0):
        action, u = self.LQR(obs, robot_idx, pos, yaw, vel, omega)
        thrust = np.random.uniform(-0.01 * self.env.M * self.env.G,
                                   0.01 * self.env.M * self.env.G)  # add noise to the thrust
        angv = np.random.uniform(-0.000001, 0.000001, 3)
        ctrl_noise = np.hstack([thrust, angv])

        u_noisy = u + ctrl_noise
        action_noisy = self.compute_low_level(u_noisy, obs[robot_idx], robot_idx)
        return action_noisy, u_noisy

    def LQR(self, obs, robot_idx, pos=np.array([0, 0, 0]), yaw=0, vel=np.array([0, 0, 0]), omega=0):
        self.lqr_controllers[robot_idx].set_desired_trajectory(robot_idx, desired_pos=pos, desired_vel=vel,
                                                               desired_acc=[0, 0, 0], desired_yaw=yaw,
                                                               desired_omega=omega)
        action, u = self.lqr_controllers[robot_idx].compute(obs)
        return action, u

    def error_state(self, x, x_des):
        e = np.copy(x)

        R_eq = Rotation.from_euler('xyz', [0, 0, x_des[2]]).as_matrix()
        R = Rotation.from_euler('xyz', x[:3]).as_matrix()
        R_err = R_eq.T @ R
        e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')
        e[6:] = R_eq.T @ (x[6:] - x_des[6:])
        e[3:6] = R_eq.T @ (x[3:6] - x_des[3:6])
        return e

    def compute_controller(self, force_diagonal=False):
        # extract the A and B matrices from the theta matrix and compute K
        m = self.m
        n = self.n
        if force_diagonal:
            self.K = np.zeros((n * self.num_robots, m * self.num_robots))
            for i in range(self.num_robots):
                A = self.theta[:m * self.num_robots, :].T[m * i:m * (i + 1), m * i:m * (i + 1)]
                B = self.theta[m * self.num_robots:, :].T[m * i:m * (i + 1), n * i:n * (i + 1)]
                # A = self.Astar[m*i:m*(i+1), m*i:m*(i+1)] testing with baseline A,B instead of learned
                # B = self.Bstar[m*i:m*(i+1), n*i:n*(i+1)]
                P = la.solve_continuous_are(A, B, self.ind_Q, self.ind_R, e=None, s=None, balanced=True)
                K_i = la.solve(self.R, B.T @ P)
                self.K[n * i:n * (i + 1), m * i:m * (i + 1)] = K_i
        else:

            A = self.theta[:m * self.num_robots, :].T  # ensure that this extracs A and B properly
            B = self.theta[m * self.num_robots:, :].T
            P = la.solve_continuous_are(A, B, self.Q, self.R, e=None, s=None, balanced=True)
            self.K = la.solve(self.R, B.T @ P)

    def set_desired_trajectory(self, robot_idx, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        self.desired_positions[robot_idx] = desired_pos
        self.desired_vels[robot_idx] = desired_vel
        self.desired_yaws[robot_idx] = desired_yaw
        self.desired_omegas[robot_idx] = desired_omega

    def compute(self, obs, skip_low_level=False):
        m = self.m
        n = self.n
        es = [self.error_state(obs_to_lin_model(obs[i], dim=m),
                               np.hstack([np.array([0, 0, self.desired_yaws[i]]), self.desired_vels[i],
                                          self.desired_positions[i]]))
              for i in range(self.num_robots)]
        # select out the correct A and B matrices for the robot
        us = np.array([-self.K[:, (m * i):m * (i + 1)] @ es[i] for i in range(self.num_robots)])
        u = np.sum(us, axis=0)

        u_robot = np.array([u[(n * i):(n * (i + 1))] for i in range(self.num_robots)])
        # offset by the equilibrium thrust
        u_robot[:, 0] += self.env.M * self.env.G
        if skip_low_level:
            #Return empty action, let the caller compute the low level control
            return None, u_robot

        action = np.array([self.compute_low_level(u, obs[idx], idx) for idx,u in enumerate(u_robot)])
        return action, u_robot

    # def cap_u(self, u):
    #     #using the min thrust based on min crazyflie rpm and max thrust
    #     u[:, 0] = np.clip(u[:, 0], 4*(9440.3**2 * self.env.KF), self.env.MAX_THRUST)
    #     return u

    def compute_low_level(self, u, obs, robot_idx):
        ctrl = self.low_level_controllers[robot_idx]
        cur_quat = obs[3:7]
        cur_ang_vel_w = obs[13:16]
        R = Rotation.from_quat(cur_quat).as_matrix()
        # convert omega from world to body frame
        cur_ang_vel_b = R.T @ cur_ang_vel_w

        action = ctrl.computeControlFromInput(u=u, control_timestep=self.env.CTRL_TIMESTEP,
                                              cur_ang_vel=cur_ang_vel_b)

        return action

    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u
