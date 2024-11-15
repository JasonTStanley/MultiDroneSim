import numpy as np
from numpy import sin, cos
from trajectories import TrajectoryBase

class CircleTrajectory(TrajectoryBase):

    def __init__(self, r=1.0, v=.5, center=np.array([0,0,0]), yaw_rate=0, revolutions=None, duration=None):
        # for now all circles are in the xy plane, constant z height = center[2]
        self.r = float(r)
        self.v = float(v)
        self.center = center
        self.yaw_rate = float(yaw_rate)

        if revolutions is not None:
            self.total_time = 2*r*np.pi*revolutions*v
        elif duration is not None:
            self.total_time = duration
        else:
            self.total_time = 2*np.pi*self.r/self.v #default to one revolution if neither duration nor revolutions are specified

    def get_total_time(self):
        return self.total_time

    def __call__(self, t):
        pos = np.zeros(3)
        vel = np.zeros(3)
        acc = np.zeros(3)
        
        yaw = (np.yaw_rate * t - np.pi) % (2*np.pi) + np.pi
        omega_yaw = np.yaw_rate

        #we start at center + r,0,0 and rotate around the z axis
        pos[0] = self.center[0] + self.r * cos(self.v/self.r * t)
        pos[1] = self.center[1] + self.r * sin(self.v/self.r * t)
        pos[2] = self.center[2]
        
        vel[0] = -self.v * sin(self.v/self.r * t)
        vel[1] = self.v * cos(self.v/self.r * t)
        vel[2] = 0

        acc[0] = -(self.v**2)/self.r * cos(self.v/self.r * t)
        acc[1] = -(self.v**2)/self.r * sin(self.v/self.r * t)
        acc[2] = 0

        return pos, vel, acc, yaw, omega_yaw


