#create an env object using params from a yaml file. this mocks the environment object from pybullet
from gym_pybullet_drones.utils.enums import DroneModel

class Environment:
    def __init__(self, G, M, MAX_THRUST, CTRL_TIMESTEP, DRONE_MODEL=DroneModel.CF2X):
        self.G = G
        self.M = M
        self.MAX_THRUST = MAX_THRUST
        self.CTRL_TIMESTEP = CTRL_TIMESTEP
        self.DRONE_MODEL = DRONE_MODEL

