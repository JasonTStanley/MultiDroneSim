import numpy as np

from trajectories import TrajectoryBase

class CompoundTrajectory(TrajectoryBase):

    def __init__(self, trajectories: list):
        '''
        :param trajectories: a list of TrajectoryBase objects
        '''
        self.trajectories = trajectories
        self.total_time = sum([t.get_total_time() for t in trajectories])
        self.times = np.cumsum([t.get_total_time() for t in trajectories])
        self.trajectory_index = 0
        self.current_trajectory = self.trajectories[self.trajectory_index]
        self.current_trajectory_time = 0

    def get_total_time(self):
        return self.total_time

    def __call__(self, t):
        if t > self.total_time:
            return self.trajectories[-1](self.trajectories[-1].get_total_time()) # final pose
        if t > self.times[self.trajectory_index]:
            self.current_trajectory_time = self.times[self.trajectory_index]
            self.trajectory_index += 1
            self.current_trajectory = self.trajectories[self.trajectory_index]

        return self.current_trajectory(t - self.current_trajectory_time)