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

class LinearizedOmegaModel:
    def __init__(self, env, debug=False):
        self.mass = env.M
        self.g = env.G
        self.A = np.zeros((9, 9))
        self.B = np.zeros((9, 4))
        self.C = np.eye(12)
        self.env = env
        self.Ahat = np.zeros((9, 9))
        self.Bhat = np.zeros((9, 4))
        self.init_matrices()


        if debug:
            print("A matrix: ")
            print(self.A)
            print("B matrix: ")
            print(self.B)




    def init_matrices(self):
        self.A[6:, 3:6] = np.eye(3)

        self.A[3, 1] = self.g
        self.A[4, 0] = -self.g

        self.B[5, 0] = 1.0 / self.mass
        self.B[:3, 1:] = np.eye(3)

        #initialize Ahat and Bhat to be a slight permutation of A and B
        self.Ahat = self.A.copy()
        self.Ahat[3, 1] = self.g*1.1
        self.Ahat[4, 0] = -self.g*1.1

        self.Bhat = self.B.copy()
        self.Bhat[5, 0] = (1.0 / self.mass) * 0.9



    def calc_xdot_from_obs(self, obs):
        '''
        :param obs: observation from the environment (includes the clipped action)
        :return: the value of x_dot for the linear model at the given observation state and action
        '''
        x = conversions.obs_to_lin_model(obs)
        action = obs[16:]
        return self.calc_xdot(x,action)

    def calc_xdot(self, x, action):
        #TODO may need to change this
        u = conversions.action_to_input(self.env, action)
        u_eq = np.array([self.mass * self.g, 0, 0, 0])
        x_eq = np.zeros((12,))
        x_eq[-3:] = x[-3:]
        x_dot = self.A @ (x - x_eq) + self.B @ (u - u_eq)

        return x_dot

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    model = LinearizedOmegaModel(debug=True)
