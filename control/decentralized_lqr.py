import numpy as np

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
        max_thrust = 4 * env.M * env.G
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
        self.lqr_controller = lqr_controller.LQRController(env, lin_models[0])
        self.K = None
        self.desired_pos = np.zeros((3,))
        self.desired_vel = np.zeros((3,))
        self.desired_yaw = 0


    def get_thetai(self, i):
        return self.theta[i * 16:(i + 1) * 16, 12 * i:(i + 1) * 12]

    def do_sim_step(self, u):
        # apply the control input to the environment
        obs = self.env.step(u)
        x_tp1 = obs_to_lin_model(obs)
        return x_tp1

    def theta_update(self, phis, xtp1s):
        # update the appropriate theta
        for i in range(self.num_robots):
            x_tp1 = xtp1s[i].reshape((12, 1))
            phi = phis[i].reshape((16, 1))
            V = self.V[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)]
            th_i = self.get_thetai(i)
            theta_new = th_i + np.linalg.inv(V) @ phi @ (x_tp1.T - phi.T @ th_i)
            V_new = V + phi @ phi.T
            self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.V[16 * i:16 * (i + 1), 16 * i:16 * (i + 1)] = V_new

    def sigma1(self):
        # draw a random input force and thrust, ensure that the input is within the bounds
        # thrust must be between 0 and 4*env.M*env.G, torques must be between -0.001 and 0.001
        thrust = np.random.uniform(.7*self.env.M*self.env.G, 1.5 * self.env.M * self.env.G)
        torques = np.random.uniform(-0.00001, 0.00001, 3)
        return np.hstack([thrust, torques])
    def LQR(self, obs, pos=np.array([0,0,0]), yaw=0, vel=np.array([0,0,0]), omega=np.array([0,0,0])):
        self.lqr_controller.set_desired_trajectory(desired_pos=pos, desired_vel=vel, desired_acc=[0,0,0], desired_yaw=yaw, desired_omega=omega)
        action = self.lqr_controller.compute(obs)
        return action

    def error_state(self, xs, xs_des):
        es = np.zeros((self.num_robots*12))
        for i in range(self.num_robots):
            x = xs[i*12:(i+1)*12]
            x_des = xs_des[i*12:(i+1)*12]
            e = np.copy(x)
            e[9:] = x[9:] - x_des[9:]
            e[6:9] = x[6:9] - x_des[6:9]
            # need to take rpy -> to rot mat -> rotate by desired yaw.T -> back to rpy
            R = Rotation.from_euler('xyz', e[0:3]).as_matrix()
            #ignore the desired pitch and roll because we don't have full freedom over them.
            Rdes = Rotation.from_euler('xyz', [0, 0, x_des[2]]).as_matrix()
            R = Rdes.T @ R
            e[0:3] = Rotation.from_matrix(R).as_euler('xyz')
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
        self.desired_pos = utils.to_ned @ desired_pos
        self.desired_vel = utils.to_ned @ desired_vel
        self.desired_yaw = -desired_yaw

    def compute(self, obs):
        x = obs_to_lin_model(obs)
        e = np.copy(x)
        e[9:] = x[9:] - self.desired_pos
        e[6:9] = (x[6:9] - self.desired_vel)

        # need to take rpy -> to rot mat -> rotate by desired yaw.T -> back to rpy
        R = Rotation.from_euler('xyz', e[0:3]).as_matrix()
        Rdes = Rotation.from_euler('xyz', [0, 0, self.desired_yaw]).as_matrix()
        R = Rdes.T @ R
        e[0:3] = Rotation.from_matrix(R).as_euler('xyz')
        u = -self.K @ e
        u[0] = -1 * u[0]  # put into body frame
        u[1:] = utils.to_ned @ u[1:]
        action = input_to_action(self.env, u)
        return action
    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u
