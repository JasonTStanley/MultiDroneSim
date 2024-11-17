import math

import numpy as np
from control.low_level.thrust_omega_ctrl import ThrustOmegaController as TOC
from scipy.spatial.transform import Rotation
from utils.model_conversions import calc_z_thrust


class YankOmegaController():
    """PID control class for Crazyflies. based on DSLPIDControl.py from the PyBullet drones library.


    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu,
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,env):
        #this is essentially a wrapper around the ThrustOmegaController that does some integration on the Yank
        #to compute a force input. (In reality we can implement a yank controller on a crazyflie as force step size over
        #some known duration)
        self.thrust_omega_ctrl = TOC(env)
        self.env = env
        #hover thrust from env
        self.hover_thrust = env.G * env.M
        self.cur_thrust = self.hover_thrust

    ################################################################################

    def reset(self):
        self.thrust_omega_ctrl.reset()
        self.cur_thrust= self.hover_thrust


    ################################################################################

    def computeControlFromInput(self, u, control_timestep, cur_ang_vel, cur_thrust):
        thrust_cmd = self.yank2thrust(u[0], control_timestep, cur_thrust)

        u_thrust = u.copy()
        u_thrust[0] = thrust_cmd
        rpm = self.thrust_omega_ctrl.computeControlFromInput(u_thrust, control_timestep, cur_ang_vel)
        return rpm



    ################################################################################

    def yank2thrust(self, yank, control_timestep, cur_thrust):
        # yank is in N/s
        # thrust is in N
        # control_timestep is in seconds
        return cur_thrust + yank * control_timestep


    ################################################################################



