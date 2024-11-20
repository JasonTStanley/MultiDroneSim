import numpy as np

from trajectories import TrajectoryBase

class RotateTrajectory(TrajectoryBase):

    def __init__(self, trajectory: TrajectoryBase, R: np.ndarray, center: np.ndarray):
        '''
        :param trajectories: a list of TrajectoryBase objects
        '''
        self.trajectory = trajectory
        self.total_time = trajectory.get_total_time()
        self.R = R
        self.center = center
        
    def get_total_time(self):
        return self.trajectory.get_total_time()

    def __call__(self, t):
        pos, vel, acc, yaw, omega = self.trajectory(t)
        R=self.R
        rot_pos = R @ (pos - self.center) + self.center
        rot_vel = R @ vel
        rot_acc = R @ acc
        return rot_pos, rot_vel, rot_acc, yaw, omega