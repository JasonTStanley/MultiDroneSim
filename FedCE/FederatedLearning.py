import numpy as np
import time
from utils import obs_to_lin_model
from typing import List
from model import LinearizedYankOmegaModel
from control.dlqr.decentralized_yolqr_crazyflie import DecentralizedYOLQRCrazyflie, YOState

class FederatedLearning:
    
    def __init__(self, env, lin_models: List[LinearizedYankOmegaModel], Q, R, P=None, num_drones=1): 
        self.num_drones = num_drones
        self.env = env
        self.linear_models = lin_models
        self.m = self.linear_models[0].m
        self.n = self.linear_models[0].n


        self.dLQR = DecentralizedYOLQRCrazyflie(env, self.linear_models, Q, R, P=P)
        self.x_prev = None

    def make_desired_state(self, pos=np.zeros(3), vel=np.zeros(3), yaw=0.0):
        #its important that the desired state has thrust = mass * gravity, as this is the equilibrium point
        return YOState(0,0, yaw, self.env.M * self.env.G ,vel[0],vel[1],vel[2],pos[0],pos[1],pos[2])

    def update(self, xtp1, x_des, u):
        # we expect that xtp1 and xdes are each a list of states for each drone, (or a [num drones, self.m] array)
        if self.x_prev is None:
            self.x_prev = xtp1
            self.x_des_prev = x_des
            self.u_prev = u
            return None, None
        
        phis = []
        e_tp1s = []

        #this could be parallelized, but this shouldn't be a bottleneck, as calculating the error states is simple
        for i in range(self.num_drones):
            e_tp1 = self.dLQR.error_state(xtp1[i], self.x_des_prev[i]) #we want to use the previous desired x here
            e_t = self.dLQR.error_state(self.x_prev[i], self.x_des_prev[i])
            u_t = self.u_prev[i]
            phis.append(np.hstack([e_t, u_t]))
            e_tp1s.append(e_tp1)

        self.x_des_prev = x_des
        self.x_prev = xtp1
        self.u_prev = u 
        self.dLQR.approx_theta_update(phis, e_tp1s, project=True)
        return phis, e_tp1s

    
    def theta_str(self):
        thetaA = self.dLQR.theta[:self.m * self.num_drones, :].T
        thetaB = self.dLQR.theta[self.m * self.num_drones:, :].T
        theta_str = "Theta A:\n"
        theta_str += np.array_str(thetaA, precision=3, suppress_small=True, max_line_width=100000)
        theta_str += "\nTheta B:\n"
        theta_str += np.array_str(thetaB, precision=3, suppress_small=True, max_line_width=100000)
        return theta_str
        
    def save_theta(self, filename="theta.npy"):
        np.save(filename, self.dLQR.theta)
    
    def lqr_control(self, x: List[YOState], x_des: List[YOState]):
        return self.dLQR.compute(x, x_des)
    
    def calc_controller(self):
        self.dLQR.compute_controller()
