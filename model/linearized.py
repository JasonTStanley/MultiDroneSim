'''
This file contains the linearized drone model from Francesco Sabatino's Thesis:
"Quadrotor Control: modeling, non-linear control design, and simulation"
(https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf)

For the linearized model see chap
The model uses state defined as in eq 2.23:
x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, vx, vy, vz, x, y, z]
x =
and input
u = [f T1 T2 T3]

The linearized model is defined as:
x_dot = Ax + Bu + Dd

where D is in R^{12x6} and is applied on the disturbance d from wind.
and d = [f_wx f_wy f_wz tau_wx tau_wy tau_wz]
'''

import numpy as np
from scipy.spatial.transform import Rotation
import utils.model_conversions as conversions

class LinearizedModel:
    def __init__(self, env, debug=False):
        self.mass = env.M
        self.Ixx = env.J[0, 0]
        self.Iyy = env.J[1, 1]
        self.Izz = env.J[2, 2]
        self.g = env.G
        self.A = np.zeros((12, 12))
        self.B = np.zeros((12, 4))
        self.C = np.eye(12)
        self.D = np.zeros((12, 6)) #dissapation matrix
        self.env = env
        self.Ahat = np.zeros((12, 12))
        self.Bhat = np.zeros((12, 4))
        self.init_matrices()


        if debug:
            print("A matrix: ")
            print(self.A)
            print("B matrix: ")
            print(self.B)

            print("D matrix: ")
            print(self.D)



    def init_matrices(self):

        self.A[0:3, 3:6] = np.eye(3)
        self.A[9:, 6:9] = np.eye(3)
        # self.A[6, 1] = -self.g
        # self.A[7, 0] = self.g

        self.A[6, 1] = self.g
        self.A[7, 0] = -self.g

        self.B[8, 0] = 1.0 / self.mass
        self.B[3:6, 1:] = np.diag([1 / self.Ixx, 1 / self.Iyy, 1 / self.Izz])

        self.D[:, 2:] = self.B.copy()
        self.D[7, 1] = 1.0 / self.mass
        self.D[6, 0] = 1.0 / self.mass

        #initialize Ahat and Bhat to be a slight permutation of A and B
        self.Ahat = self.A.copy()
        self.Ahat[6, 1] = self.g*0.9
        self.Ahat[7, 0] = -self.g*0.9

        self.Bhat = self.B.copy()
        self.Bhat[3:6, 1:] = np.array([[1 / self.Ixx, 0, 0], [0, 1 / self.Iyy, 0], [0, 0, 1 / self.Izz]]) * 1.1
        self.Bhat[8, 0] = (1.0 / self.mass) * 1.1



    def calc_xdot_from_obs(self, obs):
        '''
        :param obs: observation from the environment (includes the clipped action)
        :return: the value of x_dot for the linear model at the given observation state and action
        '''
        x = conversions.obs_to_lin_model(obs)
        action = obs[16:]
        return self.calc_xdot(x,action)

    def calc_xdot(self, x, action):
        u = conversions.action_to_input(self.env, action)
        #TODO change to standard frame instead of NED convention
        # to_ned = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # u[1:] = to_ned @ u[1:]
        # u[0] = -u[0]
        # u_eq = np.array([-self.mass * self.g, 0, 0, 0])
        u_eq = np.array([self.mass * self.g, 0, 0, 0])
        x_eq = np.zeros((12,))
        x_eq[-3:] = x[-3:]
        x_dot = self.A @ (x - x_eq) + self.B @ (u - u_eq)

        return x_dot

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    model = LinearizedModel(debug=True)
