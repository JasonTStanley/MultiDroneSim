'''
This file contains the linearized drone model from Francesco Sabatino's Thesis:
"Quadrotor Control: modeling, non-linear control design, and simulation"
(https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf)

For the linearized model see chap
The model uses state defined as in eq 2.23:
x = [roll, pitch, yaw, F, vx, vy, vz, x, y, z] where F is the current thrust force
and input
u = [Y wx wy wz]
where Y (Yank) = F dot, and wx, wy, wz are the angular velocities in the body frame.
The linearized model is defined as:
x_dot = Ax + Bu
'''

import numpy as np
from scipy.spatial.transform import Rotation
import utils.model_conversions as conversions

class LinearizedYankOmegaModel:
    def __init__(self, env, debug=False):
        self.mass = env.M
        self.g = env.G
        self.A = np.zeros((10, 10))
        self.B = np.zeros((10, 4))
        self.C = np.eye(12)
        self.env = env
        self.Ahat = np.zeros((10, 10))
        self.Bhat = np.zeros((10, 4))
        self.init_matrices()


        if debug:
            print("A matrix: ")
            print(self.A)
            print("B matrix: ")
            print(self.B)




    def init_matrices(self):
        self.A[7:, 4:7] = np.eye(3)

        self.A[4, 1] = self.g
        self.A[5, 0] = -self.g
        self.A[6,3] = 1.0 / self.mass # vz dot = 1/mass * F
        self.B[:3, 1:] = np.eye(3)
        self.B[3, 0] = 1.0 # F dot = Yank

        #initialize Ahat and Bhat to be a slight permutation of A and B
        self.Ahat = self.A.copy()
        self.Ahat[4, 1] = self.g*1.2
        self.Ahat[5, 0] = -self.g*1.2
        self.Ahat[6, 3] = (1.0 / (self.mass * 0.8))

        self.Bhat = self.B.copy()




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
    model = LinearizedYankOmegaModel(debug=True)
