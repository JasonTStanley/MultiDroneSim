class BaseController:

    def __init__(self, env):
        self.env = env

    def set_desired_trajectory(self, robot_idx, desired_pos, desired_vel, desired_acc, desired_yaw, desired_omega):
        '''Set the desired trajectory for the controller'''
        pass

    def compute(self, obs, skip_low_level=False):
        '''Given an observation in the environment, compute the control action'''
        pass
