import numpy as np
import scipy.integrate
import scipy.linalg as la
from scipy.spatial.transform import Rotation

from control.lqr import lqr_omega_controller, lqr_YO_controller
from model import LinearizedYankOmegaModel
from utils import obs_to_lin_model
from typing import List

#TODO this YO state could be defined elsewhere, but we should choose a proper location for it.

class YOState:
    
    def __init__(self, r,p,y,T,vx,vy,vz,px,py,pz):
        self.r = r
        self.p = p
        self.y = y
        self.T = T
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.px = px
        self.py = py
        self.pz = pz

    
    def set_from_idx(self, idx, value):
        if idx == 0:
            self.r = value
        elif idx == 1:
            self.p = value
        elif idx == 2:
            self.y = value
        elif idx == 3:
            self.T = value
        elif idx == 4:
            self.vx = value
        elif idx == 5:
            self.vy = value
        elif idx == 6:
            self.vz = value
        elif idx == 7:
            self.px = value
        elif idx == 8:
            self.py = value
        elif idx == 9:
            self.pz = value
        else:
            raise ValueError("Index out of bounds")

    def pos(self):
        return np.array([self.px, self.py, self.pz])
    
    def rpy(self):
        return np.array([self.r, self.p, self.y])
    
    def vel(self):
        return np.array([self.vx, self.vy, self.vz])
    
    def set_pos(self, pos):
        self.px = pos[0]
        self.py = pos[1]
        self.pz = pos[2]
    
    def set_rpy(self, rpy):
        self.r = rpy[0]
        self.p = rpy[1]
        self.y = rpy[2]
    
    def set_vel(self, vel):
        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]
    
    def set_thrust(self, thrust):
        self.T = thrust

    def __setitem__(self, index, value):
        self.set_from_idx(index, value)
    
    def __getitem__(self, index):
        return self.get_state_vec()[index]

    def copy(self):
        return YOState(self.r, self.p, self.y, self.T, self.vx, self.vy, self.vz, self.px, self.py, self.pz)
    
    def error_state(self, other):
        e = self.copy()

        R_eq = Rotation.from_euler('xyz', [0, 0, other.y]).as_matrix()
        R = Rotation.from_euler('xyz', self.rpy()).as_matrix()
        R_err = R_eq.T @ R
        e.set_rpy(Rotation.from_matrix(R_err).as_euler('xyz'))
        #TODO Will need to implement a reference thrust that is equal to hover + z_acc / mass, 
        #  for now lets say thrust error is zero always so that we keep it as a simple integrator
        # e.set_thrust(self.T - other.T)
        # e.set_thrust(0)
        e.set_thrust(self.T - other.T)
        e.set_vel(R_eq.T @ (self.vel()- other.vel()))
        e.set_pos(R_eq.T @ (self.pos() - other.pos()))
        return e
        

    def get_state_vec(self):
        return np.array([self.r,
                         self.p,
                         self.y,
                         self.T,
                         self.vx,
                         self.vy,
                         self.vz,
                         self.px,
                         self.py,
                         self.pz])


#TODO this should extend BaseController, but we must redesign the inheritence to remove reliance on sim
class DecentralizedYOLQRCrazyflie:
    def __init__(self, env, lin_models: List[LinearizedYankOmegaModel], indQ, indR, debug=False, P=None):
        from control import YankOmegaController as YOC
        # set dimensions of model
        # A is mxm B is mxn, m=9, n=4
        self.m = 10
        self.n = 4
        self.mn = self.m + self.n
        self.env = env
        self.lin_models = lin_models
        self.num_robots = len(lin_models)
        self.ind_Q = indQ
        self.ind_R = indR
        self.Q = np.kron(np.eye(self.num_robots), self.ind_Q)  # duplicate the Q and R matrices for each robot
        self.R = np.kron(np.eye(self.num_robots), self.ind_R)

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


        #TODO Tuning P matrix is important to the model convergence, definitely need to test this.
        if P is None:
            self.P = np.repeat(5*np.eye(self.mn)[:, :, None], self.num_robots, axis=2).transpose(2, 0, 1)
            # for i in range(self.num_robots):
            #     self.P[i][-3:, -3:] = 5_000_000 * np.eye(3)
        else:
            self.P = P

        self.lqr_controllers = [lqr_YO_controller.LQRYankOmegaController(env, lin_models[i], YOC(env))
                                for i in range(self.num_robots)]
        self.K = None
        self.compute_controller()
        self.desired_positions = np.zeros((self.num_robots, 3))
        self.desired_vels = np.zeros((self.num_robots, 3))
        self.desired_yaws = np.zeros(self.num_robots)
        self.desired_omegas = np.zeros(self.num_robots)

        A_keep_1 = np.index_exp[(4, 5), (1, 0)]  # refers to (6,1) and (7,0) the g values in the A matrix
        # A_keep_2 = np.index_exp[0:3, 3:6]
        A_keep_2 = np.index_exp[(7, 8, 9), (4, 5, 6)]
        # A_keep_3 = np.index_exp[9:, 6:9]
        # A_keep_3 = np.index_exp[(9, 10, 11), (6, 7, 8)]
        A_keep_3 = np.index_exp[6, 3]
    
        B_keep_1 = np.index_exp[(0,1,2), (1,2,3)]
        B_keep_2 = np.index_exp[3, 0]
        self.A_keep_2 = A_keep_2
        self.A_keep_3 = A_keep_3
        m = self.m
        n = self.n
        self.A_mask = np.zeros((m * self.num_robots, m * self.num_robots))
        self.B_mask = np.zeros((m * self.num_robots, n * self.num_robots))
        for i in range(self.num_robots):
            self.A_mask[m * i:m * (i + 1), m * i:m * (i + 1)][A_keep_1] = 1
            self.A_mask[m * i:m * (i + 1), m * i:m * (i + 1)][A_keep_2] = 1
            self.A_mask[m * i:m * (i + 1), m * i:m * (i + 1)][A_keep_3] = 1
            self.B_mask[m * i:m * (i + 1), n * i:n * (i + 1)][B_keep_1] = 1
            self.B_mask[m * i:m * (i + 1), n * i:n * (i + 1)][B_keep_2] = 1

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

    def est_x_dot(self, x_tp1, phi):
        m = self.m
        n = self.n
        # TODO experiment with how this is calculated
        phi = phi.reshape((m+n,))
        x_tp1 = x_tp1.reshape((m,))
        x_dot = np.zeros((m,))  # estimate x_dot using what we know about the system
        # x_dot[0:3] = (phi[3:6] + x_tp1[3:6]) / 2.0  # avg of the two ang vels


        #assume that the angular velocity is exactly what we applied to the system
        x_dot[0:3] = phi[m+1:] # ensure that this is the right shape
        x_dot[3] = phi[m] # thrust_dot = yank
        x_dot[4:7] = (x_tp1[4:7] - phi[4:7]) / self.env.CTRL_TIMESTEP  # estimate the linear acceleration, this can also be taken from cf
        x_dot[7:] = x_tp1[4:7] # position is an integrator
        return x_dot
    
    def project_theta(self):
        m = self.m
        n = self.n
        Ahat = self.theta[:-(n*self.num_robots), :].T
        Bhat = self.theta[-(n*self.num_robots):, :].T
        Ahat = Ahat * self.A_mask
        Bhat = Bhat * self.B_mask

        #if we need to force the values to be identity 1, for now this just forces pdot = v exactly
        for i in range(self.num_robots):
            Ahat[m * i:m * (i + 1), m * i:m * (i + 1)][self.A_keep_2] = 1
            
        self.theta = np.hstack([Ahat, Bhat]).T

    def approx_theta_update(self, phis, xtp1s, project=True):
        '''
        :param phis: list of phi vectors for each robot
        :param xtp1s: list of x_{t+1} vectors for each robot

        This method estimates the x_dot for each drone and uses that to update the system model
        '''
        for i in range(self.num_robots):
            m = self.m
            n = self.n
            # we need to estimate x_dot from observations of x_tp1 so we regress to the continuous model
            x_tp1 = xtp1s[i].reshape((m, 1))
            phi = phis[i].reshape((m+n, 1))
            x_dot = self.est_x_dot(x_tp1, phi)  # estimate x_dot

            P = self.P[i]
            L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)
            # print("L norm: " + str(np.linalg.norm(L)))
            th_i = self.get_thetai(i)
            theta_new = th_i + L @ (x_dot.T - phi.T @ th_i)
            # update the corresponding parts of theta

            # self.theta[i * 16:(i + 1) * 16, i * 12:(i + 1) * 12] = theta_new
            self.overwrite_theta(theta_new, i)
            if project:
                self.project_theta()
            self.P[i] = (np.eye(m+n) - L @ phi.T) @ P  # update P

            # Ahat = self.get_thetai(i)[:m, :].T
            # Bhat = self.get_thetai(i)[m:, :].T
            # self.pred_thetas[i].append(np.hstack([Ahat, Bhat]))

            # pred_gt_xtp1 = self.forward_predict(phi[:-4].flatten(), phi[-4:].flatten(), i, Ahat=self.lin_models[i].A,
            #                                     Bhat=self.lin_models[i].B)

            # self.pred_errors[i].append(np.linalg.norm(x_dot.T - phi.T @ th_i))
            # th_i_gt = np.hstack([self.lin_models[i].A, self.lin_models[i].B]).T
            # self.pred_errors[i + self.num_robots].append(np.linalg.norm(x_dot.T - phi.T @ th_i_gt))

    #This theta update uses x_{t+1} - x_{t+1}^hat as the error, but this doesn't work for a continuous time system, we need to use the error in xdot
    def OLD_theta_update(self, phis, xtp1s):

        for i in range(self.num_robots):
            x_tp1 = xtp1s[i].reshape((self.m, 1))
            phi = phis[i].reshape((self.mn, 1))

            P = self.P[i]
            L = P @ phi @ np.linalg.inv(1 + phi.T @ P @ phi)

            th_i = self.get_thetai(i)
            
            pred_xtp1 = self.forward_predict(phi[:-self.n].flatten(), phi[-self.n:].flatten(), i)
            #pred_xtp1 = self.forward_predict(x_tp1.flatten(), phi[-self.n:].flatten(), i)
            # pred_xtp1 = self.solve_xtp1(x_tp1, phi[-4:], i).flatten()

            theta_new = th_i + L @ (x_tp1.T - pred_xtp1)
            self.overwrite_theta(theta_new, i)
            self.P[i] = (np.eye(self.mn) - L @ phi.T) @ P  # update P


    def sigma_explore(self):
        # do gausian noise around hover with some small covariance for the input
        thrust_cov = .005 * self.env.M * self.env.G #this should use yank instead of thrust
        angv_xy_cov = 0.000000005
        angv_z_cov = 0.000000005
        thrust = np.random.normal(self.env.M * self.env.G, thrust_cov)
        angv = np.random.normal(0, [angv_xy_cov, angv_xy_cov, angv_z_cov])
        # may need to bound system. ie if we have moved too far in one direction, only allow exploration in the other direction
        return np.hstack([thrust, angv])

    def LQR(self, obs, robot_idx, pos=np.array([0, 0, 0]), yaw=0, vel=np.array([0, 0, 0]), omega=0):
        self.lqr_controllers[robot_idx].set_desired_trajectory(robot_idx, desired_pos=pos, desired_vel=vel,
                                                               desired_acc=[0, 0, 0], desired_yaw=yaw,
                                                               desired_omega=omega)
        _, u = self.lqr_controllers[robot_idx].compute(obs, skip_low_level=True)
        return u

    def error_state(self, x: YOState, x_des: YOState):
        e = x.error_state(x_des)
        return e.get_state_vec()

    def compute_controller(self):
        # extract the A and B matrices from the theta matrix and compute K
        m = self.m

        A = self.theta[:m * self.num_robots, :].T  # ensure that this extracs A and B properly
        B = self.theta[m * self.num_robots:, :].T
        P = la.solve_continuous_are(A, B, self.Q, self.R, e=None, s=None, balanced=True)
        self.K = la.solve(self.R, B.T @ P)


    def compute(self, x, x_des):
        m = self.m
        n = self.n
        es = [self.error_state(x[i], x_des[i]) for i in range(self.num_robots)]
        # select out the correct A and B matrices for the robot
        
        # u = -K @ e
        print("K shape: " + str(self.K.shape))
        us = np.array([-self.K[:, (m * i):m * (i + 1)] @ es[i] for i in range(self.num_robots)])
        u = np.sum(us, axis=0)

        u_robot = np.array([u[(n * i):(n * (i + 1))] for i in range(self.num_robots)])
        
        return u_robot

    # def cap_u(self, u):
    #     #using the min thrust based on min crazyflie rpm and max thrust
    #     u[:, 0] = np.clip(u[:, 0], 4*(9440.3**2 * self.env.KF), self.env.MAX_THRUST)
    #     return u


    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u
