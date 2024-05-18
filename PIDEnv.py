# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import argparse
import time

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
# Pybullet drone environment
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool

import MultiDroneExample
from utils import logging
import threading

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = False
DEFAULT_RECORD = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_DURATION_SEC = None # None implies run until a stop command is issued
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 2


class MultiDroneEnv(object):
    def build_args(self, kwargs):
        args = {}
        args['drone'] = DEFAULT_DRONES
        args['num_drones'] = DEFAULT_NUM_DRONES
        args['physics'] = DEFAULT_PHYSICS
        args['gui'] = DEFAULT_GUI
        args['plot'] = DEFAULT_PLOT
        args['user_debug_gui'] = DEFAULT_USER_DEBUG_GUI
        args['simulation_freq_hz'] = DEFAULT_SIMULATION_FREQ_HZ
        args['control_freq_hz'] = DEFAULT_CONTROL_FREQ_HZ
        args['duration_sec'] = None #default to no end time
        args['output_folder'] = DEFAULT_OUTPUT_FOLDER
        args['init_rad'] = 1.0
        for key, value in kwargs.items():
            if key in args:
                args[key] = value
            else:
                logging.warn(f"Skipping invalid argument: {key} in creation of MultiDroneEnv,"
                             f" must be from list {args.keys()}")
        return args

    def __init__(self, INIT_XYZS=None, INIT_RPYS=None, TARGET_POSITIONS=None, TARGET_RPYS=None, args=None, **kwargs):
        """
        :param INIT_XYZS: (num_drones, 3) np.array of initial xyz positions of drones default (1 drone at (0,0,0) the rest in a circle around it)
        :param INIT_RPYS: (num_drones, 3) np.array of initial rpy orientations of drones (default: (0,0,0)
        :param TARGET_POSITIONS: (num_drones, 3) np.array of initial target xyz positions of drones (defaults to 1m above INIT_XYZS)
        :param TARGET_RPYS: (num_drones, 3) np.array of initial target rpy orientations of drones (default: (0,0,0))
        :param args: optional arguments for the environment, will default to reasonable values
        :param kwargs: another way to specify environment args, will use keys that match args (see MultiDroneEnv.build_args)
        """
        if args is None:
            self.args = self.build_args(kwargs)
        else:
            self.args = args
        starting_target_offset = 1  # initial goal is just 1m above start pose
        # initialize the drones in a circle with radius args.init_rad centered around the first drone at (0,0,0)
        if INIT_XYZS is None:
            INIT_XYZS = np.zeros((ARGS.num_drones, 3))

            for i in range(1, ARGS.num_drones):  # first drone starts at (0,0,0) so don't initialize on circle
                INIT_XYZS[i, 0] = ARGS.init_rad * np.cos((i / ARGS.num_drones) * 2 * np.pi)
                INIT_XYZS[i, 1] = ARGS.init_rad * np.sin((i / ARGS.num_drones) * 2 * np.pi)
                INIT_XYZS[i, 2] = 0.0
        if INIT_RPYS is None:
            INIT_RPYS = np.zeros((ARGS.num_drones, 3))  # for now start all drones at Identity starting orientataion

        if TARGET_POSITIONS is None:
            # initialize the target positions and orientations to just be a vertical offset of starting positions
            TARGET_POSITIONS = np.zeros((ARGS.num_drones, 3))
            for i in range(ARGS.num_drones):
                TARGET_POSITIONS[i, 0] = INIT_XYZS[i, 0]
                TARGET_POSITIONS[i, 1] = INIT_XYZS[i, 1]
                TARGET_POSITIONS[i, 2] = INIT_XYZS[i, 2] + starting_target_offset

        if TARGET_RPYS is None:
            TARGET_RPYS = np.zeros((ARGS.num_drones, 3))

        self.INIT_XYZS = INIT_XYZS
        self.INIT_RPYS = INIT_RPYS
        self.TARGET_POSITIONS = TARGET_POSITIONS
        self.TARGET_RPYS = TARGET_RPYS
        self.ctrl = []
        self.env = None
        self.action = None
        self.obs = None
        self.stop_cmd = False

    def threaded_sim(self):
        self.stop_cmd = False
        t = threading.Thread(target=self.run_sim)
        t.start()
        return t
    def run_sim(self):
        args = self.args
        env = CtrlAviary(drone_model=args.drone,
                         num_drones=args.num_drones,
                         initial_xyzs=self.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=args.physics,
                         pyb_freq=args.simulation_freq_hz,
                         ctrl_freq=args.control_freq_hz,
                         gui=args.gui,
                         user_debug_gui=args.user_debug_gui,
                         output_folder=args.output_folder
                         )
        self.env = env
        # for information about collecting a dataset using similar code, see https://github.com/altwaitan/DL4IO/blob/main/examples/pybullet/data_collection.py
        self.PYB_CLIENT = self.env.getPyBulletClient()
        self.DRONE_IDS = self.env.getDroneIds()
        self.env._showDroneLocalAxes(0)

        # PID control for set point regulation
        ctrl = []
        if args.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            for i in range(args.num_drones):
                ctrl.append(DSLPIDControl(drone_model=args.drone))
                # Setting control gains
                ctrl[i].P_COEFF_FOR = 0.5 * np.array([.4, .4, 1.25])
                ctrl[i].I_COEFF_FOR = 0.5 * np.array([.05, .05, .05])
                ctrl[i].D_COEFF_FOR = 0.5 * np.array([.2, .2, .5])
                ctrl[i].P_COEFF_TOR = 0.5 * np.array([70000., 70000., 60000.])
                ctrl[i].I_COEFF_TOR = 0.5 * np.array([.0, .0, 500.])
                ctrl[i].D_COEFF_TOR = 0.5 * np.array([20000., 20000., 12000.])

        self.ctrl = ctrl
        # Conversion matrix between motor speeds and thrust and torques.
        r = self.env.KM / self.env.KF
        # convert between RPM and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
        conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                                   [0.0, self.env.L, 0.0, -self.env.L],
                                   [-self.env.L, 0.0, self.env.L, 0.0],
                                   [-r, r, -r, r]])

        # Run the simulation
        self.START = time.time()
        self.action = np.zeros((args.num_drones, 4))
        self.obs, _, _, _, _ = self.env.step(self.action)
        if args.duration_sec is not None:
            CTRL_STEPS = int(args.duration_sec * self.env.CTRL_FREQ)
            for i in range(CTRL_STEPS):
                self.sim_step(i)
        else:
            i = 0
            while not self.stop_sim(i):
                self.sim_step(i)
                i += 1
        # Close the environment
        self.env.close()

    def sim_step(self, i):
        args = self.args
        # The action from the PID controller is motor speed for each motor.
        for j in range(args.num_drones):
            p,v,r,w = self.obs[j][:]# find indices to get geometric controller state
            self.action[j, :], _, _ = self.ctrl[j].computeControlFromState(control_timestep=self.env.CTRL_TIMESTEP,
                                                                 state=self.obs[j],
                                                                 target_pos=self.TARGET_POSITIONS[j, :],
                                                                 target_rpy=self.TARGET_RPYS[j, :])
            # logging.info(f"Action:\n {action}")
            # logging.info(f"Obs:\n {obs[0]}")
            # logging.info(f"Target Pos:\n {TARGET_POSITIONS[0, :]}")
            # logging.info(f"Target RPY:\n {TARGET_RPYS[0, :]}")

        # Apply the control input
        self.obs, _, _, _, _ = self.env.step(self.action)

        # self.env.render()
        sync(i, self.START, self.env.CTRL_TIMESTEP)

    def stop_sim(self, i):
        if i * self.env.CTRL_TIMESTEP > 1000: #just check that the process doesn't go on forever for now
            return True
        return self.stop_cmd

    def stop(self): #async stop
        self.stop_cmd = True

if __name__ == "__main__":
    ARGS = MultiDroneExample.parse_args()
    env = MultiDroneEnv(args=ARGS)
    t = env.threaded_sim()
    #start a simulation thread
    while True:
        #set waypoints in the environment
        keyboard_cmd = input("Enter a command: ")
        if keyboard_cmd == "stop":
            print("Sending Stop Command")
            env.stop()
            break
        if keyboard_cmd.startswith("goal"):
            drone_num = int(keyboard_cmd.split()[1])
            goal_x = float(keyboard_cmd.split()[2])
            goal_y = float(keyboard_cmd.split()[3])
            goal_z = float(keyboard_cmd.split()[4])
            env.TARGET_POSITIONS[drone_num] = np.array([goal_x, goal_y, goal_z])
            print(f"Set goal for drone {drone_num} to {goal_x, goal_y, goal_z}")
        time.sleep(.01)
    t.join()
    # env = create_env(ARGS)
    # do_control(ARGS, env)
