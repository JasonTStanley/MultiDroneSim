'''
See Linearized for more details on the model, here we reduce the size of the state to match what we are able to use in
the crazyflie controller.

x = [ yaw, x,y,z, vx, vy, vz ]

and input
u = [f, pitch, roll, yaw_rate]

The linearized model is defined as:
x_dot = Ax + Bu

'''

import numpy as np
from scipy.spatial.transform import Rotation
import utils.model_conversions as conversions

class CrazyflieModel:
    def __init__(self, env, debug=False):
        self.mass = env.M
        self.g = env.G
        self.A = np.zeros((7, 7))
        self.B = np.zeros((7, 4))
        self.env = env
        self.Ahat = np.zeros((7, 7))
        self.Bhat = np.zeros((7, 4))
        self.init_matrices()


        if debug:
            print("A matrix: ")
            print(self.A)
            print("B matrix: ")
            print(self.B)




    def init_matrices(self):


        self.A[1:4, 4:7] = np.eye(3)
        G = np.array([[0, self.g], [-self.g, 0]])
        self.B[0, -1] = 1
        self.B[-1, 0] = 1.0 / self.mass
        self.B[1:3, 1:3] = G

        #initialize Ahat and Bhat to be a slight permutation of A and B
        self.Ahat = self.A.copy()

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
    class Env:
        def __init__(self):
            self.M = 0.033970
            self.G = 9.81
    env = Env()
    model = CrazyflieModel(env, debug=True)
