import numpy as np

from trajectories import TrajectoryBase
class WaitTrajectory(TrajectoryBase):
    def __init__(self, position: np.ndarray, duration: float, yaw=0):
        self.duration = duration
        self.position = position
        self.yaw = yaw

    def get_total_time(self):
        return self.duration

    def __call__(self, t):
        return self.position, np.zeros(3), np.zeros(3), self.yaw, 0
    
class LineTrajectory(TrajectoryBase):
    def __init__(self, start: np.ndarray, end: np.ndarray, speed: float = None, duration: float = None, v0=0, vf=0):
        '''
        :param start: start position xyz
        :param end: end position xyz
        :param speed: the constant speed of the trajectory in the direction of the end-start vector, in m/s (overrides duration)
        :param duration: the duration of the trajectory in seconds. If None, the duration is calculated based on the speed
        '''
        assert speed > 0, "Speed must be positive"
        if duration is not None:
            assert duration > 0, "Duration must be positive"

        self.start = start
        self.end = end
        self.delta = end - start
        self.v0 = v0
        self.vf = vf
        self.max_acc = 1.0
        if speed is None:
            self.speed = np.linalg.norm(self.delta) / duration
        else:
            self.speed = speed

        self.delta_v_init = self.speed * self.delta / np.linalg.norm(self.delta) - self.v0
        self.delta_v_end = self.vf - self.speed * self.delta / np.linalg.norm(self.delta)
        self.time_init = np.linalg.norm(self.delta_v_init) / self.max_acc
        self.time_end = np.linalg.norm(self.delta_v_end) / self.max_acc
        self.dist_init = self.v0 * self.time_init + 0.5 * self.max_acc * self.time_init ** 2
        self.dist_end = self.vf * self.time_end + 0.5 * self.max_acc * self.time_end ** 2
        self.dist_middle = np.linalg.norm(self.delta) - self.dist_init - self.dist_end
        self.time_middle = self.dist_middle / self.speed
        self.total_time = self.time_init + self.time_middle + self.time_end
    def get_total_time(self):
        return self.total_time
    def __call__(self, t):
        if t > self.total_time:
            return self.end, self.vf, np.zeros(3), 0, 0

        if t < self.time_init:
            pos = self.start + self.v0 * t + 0.5 * np.sign(self.delta_v_init) * self.max_acc * t ** 2
            vel = self.v0 + np.sign(self.delta_v_init) * self.max_acc * t
            acc = np.sign(self.delta_v_init) * self.max_acc
            yaw = 0
            omega = 0
        elif t < self.time_middle + self.time_init:
            t -= self.time_init
            delta_pos_init = self.v0 * self.time_init + 0.5 * np.sign(self.delta_v_init) * self.max_acc * self.time_init ** 2
            vel = self.speed * self.delta / np.linalg.norm(self.delta)
            pos = self.start + delta_pos_init + vel * t
            acc = np.zeros(3)
            yaw = 0
            omega = 0
        else:
            t = t - self.time_middle - self.time_init
            delta_pos_init =  self.v0 * self.time_init + 0.5 * np.sign( self.delta_v_init) * self.max_acc * self.time_init ** 2
            delta_pos_middle = delta_pos_init + (self.speed * self.delta / np.linalg.norm(self.delta)) * self.time_middle
            pos = self.start + delta_pos_middle + (self.speed * self.delta / np.linalg.norm(self.delta)) * t + 0.5 * np.sign(self.delta_v_end) * self.max_acc * t ** 2
            vel = (self.speed * self.delta / np.linalg.norm(self.delta)) + np.sign(self.delta_v_end) * self.max_acc * t
            acc = np.sign(self.delta_v_end) * self.max_acc
            yaw = 0
            omega = 0
        # vel =
        # pos = self.start + self.delta * (t / self.total_time)
        # vel = self.delta / self.total_time
        # acc = np.zeros(3)
        # yaw = 0
        # omega = 0
        return pos, vel, acc, yaw, omega