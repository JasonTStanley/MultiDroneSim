import numpy as np
import scipy.integrate

import scipy.linalg as la
from control.base_controller import BaseController
from control import lqr_controller
from model.linearized import LinearizedModel
from utils import obs_to_lin_model, input_to_action
import utils
from scipy.spatial.transform import Rotation


class DecentralizedLQR(BaseController):
    def __init__(self, env, lin_models: [LinearizedModel]):
        super().__init__(env)

        # Brysons rule, essentially set Rii to be 1/(u^2_i) where u_i is the max input for the ith value)
        max_thrust = env.MAX_THRUST
        max_torque_pitch_roll = 0.001
        max_torque_yaw = 0.001
        rflat = [1 / (max_thrust ** 2), 1 / (max_torque_pitch_roll ** 2), 1 / (max_torque_pitch_roll ** 2),
                 1 / (max_torque_yaw ** 2)]
        R = np.diag(rflat)
        max_vel_error = .15
        max_pos_error = .05
        max_yaw_error = np.pi / 40
        max_pitch_roll_error = np.pi / 40
        max_pitch_yaw_rate_error = .25
        max_yaw_rate_error = .25
        # stack into Q in order of state x=[r, p, y, r_dot, p_dot, y_dot, vx, vy, vz, px, py, pz]
        qflat = [1 / (max_pitch_roll_error ** 2), 1 / (max_pitch_roll_error ** 2), 1 / (max_yaw_error ** 2),
                 1 / (max_pitch_yaw_rate_error ** 2), 1 / (max_pitch_yaw_rate_error ** 2),
                 1 / (max_yaw_rate_error ** 2),
                 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2), 1 / (max_vel_error ** 2),
                 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2), 1 / (max_pos_error ** 2)]
        Q = np.diag(qflat)

        print(qflat)
        print(rflat)
        self.lin_models = lin_models
        self.Q = Q
        self.R = R
        self.num_robots = len(lin_models)
        self.Astar = np.zeros((12 * self.num_robots, 12 * self.num_robots))
        self.Bstar = np.zeros((12 * self.num_robots, 4 * self.num_robots))
        for i, agent in enumerate(lin_models):
            assert agent.Ahat.shape == (12, 12)
            assert agent.Bhat.shape == (12, 4)
            # insert the agent's approximate A and B into the correct position in the Astar and Bstar matrices
            self.Astar[i * 12:(i + 1) * 12, i * 12:(i + 1) * 12] = agent.Ahat
            self.Bstar[i * 12:(i + 1) * 12, i * 4:(i + 1) * 4] = agent.Bhat
            #here Ahat is the guess of the A matrix of the linearized model

        self.theta = np.hstack([self.Astar, self.Bstar]).T
        self.V = np.eye(16 * self.num_robots)
        self.P = np.eye(16 * self.num_robots)
        self.lqr_controller = lqr_controller.LQRController(env, lin_models[0])
        self.K = None
        self.desired_pos = np.zeros((3,))
        self.desired_vel = np.zeros((3,))
        self.desired_yaw = 0
        self.desired_omega = 0
        A_keep_1 = np.index_exp[(6, 7), (1, 0)]  # refers to (6,1) and (7,0) the g values in the A matrix
        # A_keep_2 = np.index_exp[0:3, 3:6]
        A_keep_2 = np.index_exp[(0,1,2), (3,4,5)]
        # A_keep_3 = np.index_exp[9:, 6:9]
        A_keep_3 = np.index_exp[(9,10,11), (6,7,8)]
        B_keep_1 = np.index_exp[3:6, 1:]
        B_keep_2 = np.index_exp[8, 0]
        self.A_mask = np.zeros((12 * self.num_robots, 12 * self.num_robots))
        self.B_mask = np.zeros((12 * self.num_robots, 4 * self.num_robots))
        for i in range(self.num_robots):
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_1] = 1
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_2] = 1
            self.A_mask[12 * i:12 * (i + 1), 12 * i:12 * (i + 1)][A_keep_3] = 1
            self.B_mask[12 * i:12 * (i + 1), 4 * i:4 * (i + 1)][B_keep_1] = 1
            self.B_mask[12 * i:12 * (i + 1), 4 * i:4 * (i + 1)][B_keep_2] = 1


    def get_thetai(self, i):
        return self.theta[i * 16:(i + 1) * 16, 12 * i:(i + 1) * 12]

    def forward_predict(self, e, u):
        Ahat = self.get_thetai(0)[:12, :].T
        Bhat = self.get_thetai(0)[12:, :].T
        def f(t, e):
            return Ahat @ e + Bhat @ u
        y0 = e
        sol = scipy.integrate.solve_ivp(f, [0, self.env.CTRL_TIMESTEP], y0)
        return sol.y[:, -1]

    def theta_update(self, phis, xtp1s):
        # update the appropriate theta
        for i in range(self.num_robots):
            #we need to estimate x_dot from observations of x_tp1 so we regress to the continuous model
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))
            x_dot = (x_tp1 - phi[:12]) / self.env.CTRL_TIMESTEP #estimate x_dot
            V = self.V[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)]
            th_i = self.get_thetai(i)
            theta_new = th_i + np.linalg.pinv(V) @ phi @ (x_dot.T - phi.T @ th_i)
            V_new = V + phi @ phi.T
            self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.V[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)] = V_new

        #project back to known zeros
        self.project_theta()

    def est_x_dot(self, x_tp1, phi):
        phi = phi.reshape((16,))
        x_tp1 = x_tp1.reshape((12,))
        x_dot = np.zeros((12,)) # estimate x_dot using what we know about the system
        x_dot[0:3] = (phi[3:6] + x_tp1[3:6]) / 2.0 # avg of the two ang vels
        x_dot[3:6] = (x_tp1[3:6] - phi[3:6]) / self.env.CTRL_TIMESTEP # estimate the angular acceleration

        x_dot[6:9] = (x_tp1[6:9] - phi[6:9]) / self.env.CTRL_TIMESTEP # estimate the linear acceleration
        x_dot[9:] = (phi[6:9] + x_tp1[6:9]) / 2.0 # avg of the two linear vels

        return x_dot
    def new_theta_update(self, phis, xtp1s):
        A_gs = np.index_exp[(6, 7), (1, 0)]
        Ahat = self.theta[:-4, :].T
        Bhat = self.theta[-4:, :].T
        ag_before = Ahat[A_gs]
        ag_expected = np.array([9.81, -9.81])

        for i in range(self.num_robots):
            # we need to estimate x_dot from observations of x_tp1 so we regress to the continuous model
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))
            x_dot = self.est_x_dot(x_tp1, phi)  # estimate x_dot

            P = self.P[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)]
            L = self.P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)
            # print("L norm: " + str(np.linalg.norm(L)))
            th_i = self.get_thetai(i)
            theta_new = th_i + L @ (x_dot.T - phi.T @ th_i)
            self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.P[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)] = (np.eye(16) - L @ phi.T) @ P # update P

        self.project_theta()
        ag_after = self.theta[:-4, :].T[A_gs]
        if np.linalg.norm(ag_after - ag_expected) > np.linalg.norm(ag_before - ag_expected) + .01:
            print("A_g not updated correctly")
            print("A_g before: " + str(ag_before))
            print("A_g after: " + str(ag_after))
            print("A_g expected: " + str(ag_expected))

    def sigma1(self):
        # draw a random input force and thrust, ensure that the input is within the bounds
        # thrust must be between 0 and 4*env.M*env.G, torques must be between -0.001 and 0.001
        thrust = np.random.uniform(.7*self.env.M*self.env.G, 1.5 * self.env.M * self.env.G)
        torques = np.random.uniform(-0.00001, 0.00001, 3)
        return np.hstack([thrust, torques])

    def sigma_explore(self):
        #do gausian noise around hover with some small covariance for the input
        thrust_cov = .15 * self.env.M * self.env.G
        torque_xy_cov = 0.005 * self.env.MAX_XY_TORQUE
        torque_z_cov = 0.005 * self.env.MAX_Z_TORQUE
        thrust = np.random.normal(self.env.M * self.env.G, thrust_cov)
        torques = np.random.normal(0, [torque_xy_cov, torque_xy_cov, torque_z_cov])
        #may need to bound system. ie if we have moved too far in one direction, only allow exploration in the other direction
        return np.hstack([thrust, torques])

    def noisy_control(self, obs):
        action , u = self.compute(obs)
        thrust = np.random.uniform(.01 * self.env.M * self.env.G, 0.01 * self.env.M * self.env.G) # add noise to the thrust
        torques = np.random.uniform(-0.000001, 0.000001, 3) # add noise to the torques
        ctrl_noise = np.hstack([thrust, torques])

        u_noisy = u + ctrl_noise
        action_noisy = input_to_action(self.env, u_noisy)
        return action_noisy, u_noisy


    def LQR(self, obs, pos=np.array([0,0,0]), yaw=0, vel=np.array([0,0,0]), omega=0):
        self.lqr_controller.set_desired_trajectory(desired_pos=pos, desired_vel=vel, desired_acc=[0,0,0], desired_yaw=yaw, desired_omega=omega)
        action = self.lqr_controller.compute(obs)
        return action

    def error_state(self, xs, xs_des):
        es = np.zeros((self.num_robots*12))
        for i in range(self.num_robots):
            x = xs[i*12:(i+1)*12]
            x_des = xs_des[i*12:(i+1)*12]
            e = np.copy(x)

            R_eq = Rotation.from_euler('xyz', [0, 0, x_des[2]]).as_matrix()
            R = Rotation.from_euler('xyz', x[:3]).as_matrix()
            R_err = R_eq.T @ R
            e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')
            e[9:] = R_eq.T @ (x[9:] - x_des[9:])
            e[6:9] = R_eq.T @ (x[6:9] - x_des[6:9])
            e[3:6] = R_eq.T @ (x[3:6] - x_des[3:6])
            es[i*12:(i+1)*12] = e
        return es

    def compute_controller(self):
        #extract the A and B matrices from the theta matrix and compute K
        A = self.theta[:12, :].T #ensure that this extracs A and B properly
        B = self.theta[12:, :].T
        P = la.solve_continuous_are(A, B, self.Q, self.R, e=None, s=None, balanced=True)

        self.K = la.solve(self.R, B.T @ P)


    def set_desired_trajectory(self, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        # offset pos by desired_pos for now
        self.desired_pos = desired_pos
        self.desired_vel = desired_vel
        self.desired_yaw = desired_yaw
        self.desired_omega = desired_omega

    def project_theta(self):
        #set the known zeros to zero

        Ahat = self.theta[:-4, :].T
        Bhat = self.theta[-4:, :].T
        Ahat = Ahat * self.A_mask
        Bhat = Bhat * self.B_mask
        Ahat[0:3, 3:6] = np.eye(3)
        Ahat[9:, 6:9] = np.eye(3)
        self.theta = np.hstack([Ahat, Bhat]).T


    def compute(self, obs):
        x = obs_to_lin_model(obs)
        e = np.copy(x)

        # equilibrium is at the desired yaw
        R_eq = Rotation.from_euler('xyz', [0, 0, self.desired_yaw]).as_matrix()
        R = Rotation.from_euler('xyz', x[:3]).as_matrix()
        R_err = R_eq.T @ R  # rotate R by the desired yaw this is the error in rotation from the equilibrium

        e[:3] = Rotation.from_matrix(R_err).as_euler('xyz')

        # since observation and desired values are in the world frame we must rotate them
        # by the desired angle (goal frame) to compute the error between body and goal
        e[9:] = R_eq.T @ (x[9:] - self.desired_pos)
        e[6:9] = R_eq.T @ (x[6:9] - self.desired_vel)
        e[3:6] = R_eq.T @ (x[3:6] - np.array([0, 0, self.desired_omega]))

        u = -self.K @ e
        #offset by the equilibrium thrust
        u[0] += self.env.M * self.env.G
        action = input_to_action(self.env, u)
        return action, u

    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u
