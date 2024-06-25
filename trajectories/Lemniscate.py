import numpy as np
from numpy import sin, cos
class TrajectoryBase:
    def __call__(self, t):
        '''
        :param t: the time to evaluate the trajectory at
        :return: the position, velocity, acceleration, and yaw angle at time t
        '''
        raise NotImplementedError

class Lemniscate(TrajectoryBase):

    def __init__(self, a=1, omega=.5, center=np.array([0,0,0]), yaw_rate=0):
        self.a = a
        self.omega = omega
        self.center = center
        self.yaw_rate = yaw_rate

    def __call__(self, t):
        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        th = t * self.omega
        # target_position.x() = center.x() + (a * sin(th) * cos(th)) / (1 + sin(th) * sin(th));
        # target_position.y() = center.y() + (a * cos(th)) / (1 + sin(th) * sin(th));
        # target_position.z() = center.z() + 0;
        # target_velocity.x() = -a * omega * (pow(sin(th), 4) + pow(sin(th), 2) + (pow(sin(th), 2) - 1) * pow(cos(th), 2)) / pow(
        #     pow(sin(th), 2) + 1, 2);
        # target_velocity.y() = -a * omega * sin(th) * (pow(sin(th), 2) + 2 * pow(cos(th), 2) + 1) / (
        #     pow(sin(th) * sin(th) + 1, 2));
        # target_velocity.z() = 0.0;
        # target_acceleration.x() = 4 * a * pow(omega, 2) * sin(2 * th) * (3 * cos(2 * th) + 7) / pow(cos(2 * th) - 3, 3);
        # target_acceleration.y() = a * pow(omega, 2) * cos(th) * (44 * cos(2 * th) + cos(4 * th) - 21) / pow(cos(2 * th) - 3, 3);
        # target_acceleration.z() = 0.0;

        pos[0] = self.center[0] + (self.a * sin(th) * cos(th)) / (1 + sin(th) ** 2)
        pos[1] = self.center[1] + (self.a * cos(th)) / (1 + sin(th) ** 2)
        pos[2] = self.center[2]

        vel[0] = -self.a * self.omega * (sin(th) ** 4 + sin(th) ** 2 + (sin(th) ** 2 - 1) * cos(th) ** 2) / (sin(th) ** 2 + 1) ** 2
        vel[1] = -self.a * self.omega * sin(th) * (sin(th) ** 2 + 2 * cos(th) ** 2 + 1) / (sin(th) ** 2 + 1) ** 2
        vel[2] = 0

        acc[0] = 4 * self.a * self.omega ** 2 * sin(2 * th) * (3 * cos(2 * th) + 7) / (cos(2 * th) - 3) ** 3
        acc[1] = self.a * self.omega ** 2 * cos(th) * (44 * cos(2 * th) + cos(4 * th) - 21) / (cos(2 * th) - 3) ** 3
        acc[2] = 0
        yaw = np.pi * sin(self.yaw_rate * t) #yaw rate smoothly changes from -pi to pi based on yaw rate
        omega = np.pi * self.yaw_rate * cos(self.yaw_rate*t)
        return pos, vel, acc, yaw, omega



