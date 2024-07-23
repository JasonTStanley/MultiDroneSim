import math

import numpy as np
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from scipy.spatial.transform import Rotation


class ThrustOmegaController(BaseControl):
    """PID control class for Crazyflies. based on DSLPIDControl.py from the PyBullet drones library.


    Based on work conducted at UTIAS' DSL. Contributors: SiQi Zhou, James Xu,
    Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,env):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        drone_model = env.DRONE_MODEL
        g = env.G
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        #try 7500, 7500, 7500 because it makes the error very low
        self.P_COEFF_OMEGA_TOR = np.array([25000., 25000., 25000.])
        self.I_COEFF_OMEGA_TOR = np.array([10., 10., 15.])

        self.D_COEFF_OMEGA_TOR = np.array([0., 0., 0.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],
                [-.5, .5, 1],
                [.5, .5, -1],
                [.5, -.5, 1]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],
                [+1, 0, 1],
                [0, 1, -1],
                [-1, 0, 1]
            ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last omega ###################
        self.last_omega = np.zeros(3)
        #### Initialized PID control variables #####################
        self.integral_omega_e = np.zeros(3)



    ################################################################################

    def computeControlFromInput(self, u, control_timestep, cur_ang_vel):
        '''
        A low level controller that resembles what would run on the flight controller.
        This computes the control input for the drone based on an input of thrust and angular velocity.

        :param u: [1d thrust, wx, wy, wz] omega must be in body frame (default from observations is world)
        :return: action: [rpm1, rpm2, rpm3, rpm4]
        '''

        self.control_counter += 1

        # ensure thrust is positive
        u[0] = np.clip(u[0], 0, None)
        pwm_thrust = np.clip((np.sqrt(u[0] / (self.KF * 4)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE, self.MIN_PWM,
                             self.MAX_PWM)

        rpm = self.omega_PID(control_timestep, pwm_thrust, cur_ang_vel, u[1:])
        return rpm


    ################################################################################

    def omega_PID(self,
                  control_timestep,
                  thrust,
                  cur_omega,
                  target_omega
                  ):

        # target angular accel is always 0 for now
        omega_rate_e = np.zeros(3) - (cur_omega - self.last_omega) / control_timestep
        omega_e = target_omega - cur_omega
        self.last_omega = cur_omega

        #maybe implement if we want to have a nonzero I term, for now leave as zero
        self.integral_omega_e = self.integral_omega_e - omega_e * control_timestep
        self.integral_omega_e = np.clip(self.integral_omega_e, -1500., 1500.)
        self.integral_omega_e[0:2] = np.clip(self.integral_omega_e[0:2], -1., 1.)

        #### PID target torques ####################################

        target_torques_omega = np.multiply(self.P_COEFF_OMEGA_TOR, omega_e) \
                            + np.multiply(self.I_COEFF_OMEGA_TOR, self.integral_omega_e) \
                           + np.multiply(self.D_COEFF_OMEGA_TOR, omega_rate_e)
        #unsure if the D gain works well, needs testing


        target_torques = target_torques_omega #potentially add a term here for balancing to compensate for lag in omega
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST


