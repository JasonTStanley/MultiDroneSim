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

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_RECORD = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_DURATION_SEC = 30
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 2


def parse_args():
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Test flight script using SITL Betaflight')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel,
                        help='Drone model (default: BETA)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int,
                        help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--init_rad', default=1.0, type=float,
                        help='Initial radius of the drones (default: 1.0)', metavar='')
    ARGS = parser.parse_args()
    return ARGS


def create_env(args):
    env = CtrlAviary(drone_model=args.drone,
                     num_drones=args.num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=args.physics,
                     pyb_freq=args.simulation_freq_hz,
                     ctrl_freq=args.control_freq_hz,
                     gui=args.gui,
                     user_debug_gui=args.user_debug_gui,
                     output_folder=args.output_folder
                     )
    return env


def do_control(args, env):
    # for information about collecting a dataset using similar code, see https://github.com/altwaitan/DL4IO/blob/main/examples/pybullet/data_collection.py
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()
    env._showDroneLocalAxes(0)

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

    # Conversion matrix between motor speeds and thrust and torques.
    r = env.KM / env.KF
    # convert between RPM and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
    conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                               [0.0, env.L, 0.0, -env.L],
                               [-env.L, 0.0, env.L, 0.0],
                               [-r, r, -r, r]])

    # Run the simulation
    START = time.time()
    action = np.zeros((args.num_drones, 4))
    obs, _, _, _, _ = env.step(action)
    CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)
    for i in range(CTRL_STEPS):
        # The action from the PID controller is motor speed for each motor.
        for j in range(args.num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos=TARGET_POSITIONS[j, :],
                                                                 target_rpy=TARGET_RPYS[j, :])
            # logging.info(f"Action:\n {action}")
            # logging.info(f"Obs:\n {obs[0]}")
            # logging.info(f"Target Pos:\n {TARGET_POSITIONS[0, :]}")
            # logging.info(f"Target RPY:\n {TARGET_RPYS[0, :]}")

        # Apply the control input
        obs, _, _, _, _ = env.step(action)

        env.render()
        sync(i, START, env.CTRL_TIMESTEP)
    # Close the environment
    env.close()


if __name__ == "__main__":
    ARGS = parse_args()
    starting_target_offset = 1  # initial goal is just 1m above start pose
    # initialize the drones in a circle with radius args.init_rad centered around the first drone at (0,0,0)
    INIT_XYZS = np.zeros((ARGS.num_drones, 3))

    for i in range(1, ARGS.num_drones):  # first drone starts at (0,0,0) so don't initialize on circle
        INIT_XYZS[i, 0] = ARGS.init_rad * np.cos((i / ARGS.num_drones) * 2 * np.pi)
        INIT_XYZS[i, 1] = ARGS.init_rad * np.sin((i / ARGS.num_drones) * 2 * np.pi)
        INIT_XYZS[i, 2] = 0.0

    # initialize the target positions and orientations to just be a vertical offset of starting positions
    TARGET_POSITIONS = np.zeros((ARGS.num_drones, 3))
    for i in range(ARGS.num_drones):
        TARGET_POSITIONS[i, 0] = INIT_XYZS[i, 0]
        TARGET_POSITIONS[i, 1] = INIT_XYZS[i, 1]
        TARGET_POSITIONS[i, 2] = INIT_XYZS[i, 2] + starting_target_offset

    INIT_RPYS = np.zeros((ARGS.num_drones, 3))  # for now start all drones at Identity starting orientataion
    TARGET_RPYS = np.zeros((ARGS.num_drones, 3))

    env = create_env(ARGS)
    do_control(ARGS, env)
