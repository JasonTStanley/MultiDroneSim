# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import argparse
import time

import numpy as np
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from scipy.spatial.transform import Rotation

import utils
import utils.model_conversions as conversions
# Pybullet drone environment
from control import GeometricControl, LQRController, DecentralizedLQR
from model import LinearizedModel
from trajectories import Lemniscate
from utils import obs_to_lin_model

DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True  # usually true
DEFAULT_PLOT = False  # usually true
DEFAULT_RECORD = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 3  # 2
controllers = ['lqr', 'geometric']  # whichever is first will be default


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
                        help=f'Simulation frequency in Hz (default: {DEFAULT_SIMULATION_FREQ_HZ})', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help=f'Control frequency in Hz (default: {DEFAULT_CONTROL_FREQ_HZ})', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--init_rad', default=1.0, type=float,
                        help='Initial radius of the drones (default: 1.0)', metavar='')
    parser.add_argument('--controller', default=controllers[0], type=str,
                        help=f'Controller to use from {controllers} (default: {controllers[0]})', metavar='')
    ARGS = parser.parse_args()
    return ARGS


class GeometricEnv:
    def __init__(self, args, circle_init=True):
        self.env = None
        self.obs = None
        self.conversion_mat = None
        self.observations = []
        self.args = args
        self.INIT_XYZS = np.zeros((args.num_drones, 3))
        self.INIT_RPYS = np.zeros((args.num_drones, 3))  # for now start all drones at Identity starting orientataion
        self.TARGET_POSITIONS = np.zeros((args.num_drones, 3))
        self.TARGET_RPYS = np.zeros((args.num_drones, 3))
        self.obs_ts = []
        self.linear_models = None

        if circle_init:
            self.starting_target_offset = 1  # initial goal is just 1m above start pose
            self.circle_initialize()

    def create_env(self, gui=True):
        args = self.args
        env = CtrlAviary(drone_model=args.drone,
                         num_drones=args.num_drones,
                         initial_xyzs=self.INIT_XYZS,
                         initial_rpys=self.INIT_RPYS,
                         physics=args.physics,
                         pyb_freq=args.simulation_freq_hz,
                         ctrl_freq=args.control_freq_hz,
                         gui=args.gui and gui,
                         user_debug_gui=args.user_debug_gui,
                         output_folder=args.output_folder
                         )
        self.env = env

        self.linear_models = [LinearizedModel(env) for _ in range(args.num_drones)]

        r = self.env.KM / self.env.KF
        # convert between motor_thrusts and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
        self.conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                                        [0.0, self.env.L, 0.0, -self.env.L],
                                        [-self.env.L, 0.0, self.env.L, 0.0],
                                        [-r, r, -r, r]])
        return self.env

    def fedCE(self):
        num_iter = 100 #number loops
        env = self.env
        args = self.args
        steps = 0
        dLQR = DecentralizedLQR(env, self.linear_models)
        START = time.time()
        for n in range(num_iter):
            steps = self.fedCE_iteration(env, dLQR, START, steps, n, do_warmup=(n==0), random_warmup=True)
            thetaA = dLQR.theta[:12*args.num_drones, :].T
            thetaB = dLQR.theta[12*args.num_drones:, :].T
            with np.printoptions(precision=3, suppress=True, linewidth=100000):
                print(f"n: {n}, steps: {steps}")
                print("Theta A:\n ", thetaA)
                print("Theta B:\n ", thetaB)

        env.close()

    def fedCE_iteration(self, env, dLQR, START, steps, n, k=2, do_warmup=True, random_warmup=True, do_lemniscate=False, do_print=False):
        Texp = n * k
        Tce = n * (k**3)
        Tw = 0
        if do_warmup:
            Tw = 25 # some small warmup

        args = self.args
        action = np.zeros((args.num_drones, 4))
        obs, _, _, _, _ = env.step(action)
        # CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)

        t = 0
        traj = Lemniscate(center=np.array([0, 0, .5]), omega=1, yaw_rate=.1)
        #warm up
        for i in range(Tw):
            phis = []
            e_tp1s = []
            x_des = np.zeros((12,))
            action = np.zeros((args.num_drones, 4))


            for j in range(args.num_drones):
                x = obs_to_lin_model(obs[j])

                if random_warmup:
                    u = dLQR.sigma1()
                    act = conversions.input_to_action(env, u)
                    # set x_des to be the starting point
                    x_des = np.zeros((12,))
                    x_des[0:3] = self.INIT_RPYS[j, :]
                    x_des[-3:] = self.INIT_XYZS[j, :]
                else:
                    pos, vel, acc, yaw, omega = traj(t)
                    x_des = np.hstack([[0, 0, yaw], [0, 0, omega], vel, pos])
                    act, u_orig = dLQR.LQR(obs[j], pos=pos, vel=vel, yaw=yaw, omega=omega)
                    u = utils.action_to_input(env, act) # clip u to be within bounds


                # offset u by the equilibrium point
                u[0] = u[0] - env.M * env.G  # TODO replace with estimate? and apply for all drones
                e = dLQR.error_state(x, x_des)
                # since we have a setpoint, we must calculate the error state to learn theta
                action[j, :] = act
                phis.append(np.hstack([e, u]))  # use error state as input state to update phi

            obs, _, _, _, _ = env.step(action)

            self.obs = obs
            t += env.CTRL_TIMESTEP

            for j in range(args.num_drones):
                # need to offset by the error
                x_tp1 = obs_to_lin_model(obs[j])
                x_des = np.zeros((12,))
                x_des[0:3] = self.INIT_RPYS[j, :]
                x_des[-3:] = self.INIT_XYZS[j, :]
                e_tp1 = dLQR.error_state(x_tp1, x_des)
                # e = phis[j][:12]
                # u = phis[j][12:]
                # pred_e_tp1 = dLQR.forward_predict(e,u)
                # print("e_tp1: ", e_tp1)
                # print("pred_e_tp1: ", pred_e_tp1)
                # print("diff: ", e_tp1 - pred_e_tp1)
                e_tp1s.append(e_tp1)

            # update the theta and V values
            if i != 0: #skip first because the control input was not applied
                # dLQR.theta_update(phis, e_tp1s)
                dLQR.new_theta_update(phis, e_tp1s)

            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1

        t = 0
        #CE phase
        last_desired = np.zeros((args.num_drones, 12))
        dLQR.compute_controller()
        for i in range(Tce):
            #compute controller from current observations and use it to achieve some task
            for j in range(args.num_drones):
                if do_lemniscate:
                    pos, vel, acc, yaw, omega = traj(t)
                    dLQR.set_desired_trajectory(j, desired_pos=pos, desired_vel=vel, desired_acc=acc,
                                                   desired_yaw=yaw,
                                                   desired_omega=omega)
                else:
                    dLQR.set_desired_trajectory(j, desired_pos=self.TARGET_POSITIONS[j, :],
                                                   desired_vel=np.zeros((3,)),
                                                   desired_acc=np.zeros((3,)),
                                                   desired_yaw=self.TARGET_RPYS[j, :][2],
                                                   desired_omega=0)
                    last_desired[j, :] = np.hstack([self.TARGET_RPYS[j, :], np.zeros((3,)), np.zeros((3,)),
                                                    self.TARGET_POSITIONS[j, :]])

            action, u = dLQR.compute(obs)

            obs, _, _, _, _ = env.step(action)
            t += env.CTRL_TIMESTEP
            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1

        #do exploration
        for i in range(Texp):
            phis = []
            e_tp1s = []
            action = np.zeros((args.num_drones, 4))
            # x_des = np.zeros((12,))
            # pos, vel, acc, yaw, omega = traj(t)
            # dLQR.set_desired_trajectory(desired_pos=pos, desired_vel=vel, desired_acc=acc, desired_yaw=yaw,
            #                             desired_omega=omega)
            # x_des = np.hstack([[0, 0, yaw], [0, 0, omega], vel, pos])
            for j in range(args.num_drones):
                x = obs_to_lin_model(obs[j])
                x_des = last_desired[j, :]
                e = dLQR.error_state(x, x_des)

                u = dLQR.sigma_explore()
                act = conversions.input_to_action(env, u)#convert to action and back to ensure we have valid control
                # act, _ = dLQR.noisy_control(obs[j]) # continue LQR control but add noise to the control input to explore states
                u = utils.action_to_input(env, act)  # clip u to be within bounds
                # offset u by the equilibrium point
                u[0] = u[0] - env.M * env.G  # TODO replace with estimate? and apply for all drones

                # since we have a setpoint, we must calculate the error state to learn theta
                action[j, :] = act
                phis.append(np.hstack([e, u]))  # use error state as input state to update phi

            obs, _, _, _, _ = env.step(action)

            for j in range(args.num_drones):
                # need to offset by the error
                x_des = last_desired[j, :]
                x_tp1 = obs_to_lin_model(obs[j])
                e_tp1 = dLQR.error_state(x_tp1, x_des)
                e_tp1s.append(e_tp1)


            # update the theta and V values
            if i != 0: #skip first because the control input was different
                # dLQR.theta_update(phis, e_tp1s)
                dLQR.new_theta_update(phis, e_tp1s)

            t += env.CTRL_TIMESTEP
            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1
        return steps



    def warm_up_only(self, TW=500):
        env = self.env
        args = self.args
        env._showDroneLocalAxes(0)


        # Run the simulation
        START = time.time()
        action = np.zeros((args.num_drones, 4))
        obs, _, _, _, _ = env.step(action)
        # CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)

        t = 0
        dLQR = DecentralizedLQR(env, self.linear_models)
        traj = Lemniscate(center=np.array([0, 0, .5]), omega=.5, yaw_rate=1)

        for i in range(TW):
            phis = []
            e_tp1s = []
            action = np.zeros((args.num_drones, 4))

            # pos, vel, acc, yaw, omega = traj(t)
            pos, vel, acc, yaw, omega = np.zeros((3,)),np.zeros((3,)),np.zeros((3,)),0,0 #when doing random warmup
            x_des = np.hstack([[0, 0, yaw], [0, 0, omega], vel, pos])
            for j in range(args.num_drones):
                x = obs_to_lin_model(obs[j])
                u = dLQR.sigma1()
                act = conversions.input_to_action(env, u)
                e = dLQR.error_state(x, x_des)
                # act, u_orig = dLQR.LQR(obs[j], pos=pos, vel=vel, yaw=yaw, omega=omega)
                # clip u to be within bounds
                u = utils.action_to_input(env, act)
                # offset u by the equilibrium point
                u[0] = u[0] - env.M * env.G  # for now use groundtruth here TODO replace with estimate?

                # since we have a setpoint, we must calculate the error state to learn theta
                action[j, :] = act
                phis.append(np.hstack([e, u]))  # use error state as input state to update phi

            obs, _, _, _, _ = env.step(action)
            self.obs = obs
            if t >= 1:
                print("t>=1")
            self.observations.append(obs)
            self.obs_ts.append(t)
            t += env.CTRL_TIMESTEP

            for j in range(args.num_drones):
                # need to offset by the error

                x_tp1 = obs_to_lin_model(obs[j])
                e_tp1 = dLQR.error_state(x_tp1, x_des)
                e_tp1s.append(e_tp1)

            # update the theta and V values
            dLQR.theta_update(phis, e_tp1s)

            env.render()
            sync(i, START, env.CTRL_TIMESTEP)

            # for j in range(args.num_drones):
            #     ctrl[j].set_desired_trajectory(desired_pos=self.TARGET_POSITIONS[j, :], desired_vel=np.zeros((3,)),
            #                                    desired_acc=np.zeros((3,)), desired_yaw=self.TARGET_RPYS[j, :][2],
            #                                    desired_omega=0)
            #     action[j, :] = ctrl[j].compute(obs[j])

        # compare theta to the true linear model
        # print("Theta: ", dLQR.theta)
        thetaA = dLQR.theta[:12, :].T
        thetaB = dLQR.theta[12:, :].T
        with np.printoptions(precision=3, suppress=True):
            print("Theta A:\n ", thetaA)
            print("Theta B:\n ", thetaB)
            print("True A0:\n ", self.linear_models[0].A)
            print("True B0:\n ", self.linear_models[0].B)
        # Close the environment
        env.close()
        dLQR.compute_controller()
        return dLQR.K, dLQR.theta

    def do_control(self, do_lemniscate=False):
        env = self.env
        args = self.args
        # for information about collecting a dataset using similar code, see https://github.com/altwaitan/DL4IO/blob/main/examples/pybullet/data_collection.py
        PYB_CLIENT = env.getPyBulletClient()
        PYB_CLIENT = env.getPyBulletClient()
        DRONE_IDS = env.getDroneIds()
        env._showDroneLocalAxes(0)

        # PID control for set point regulation
        ctrl = []
        traj = Lemniscate(center=np.array([0, 0, .5]), omega=0.5, yaw_rate=0.2)
        if args.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            for i in range(args.num_drones):
                if args.controller == "geometric":
                    geo_ctrl = GeometricControl(env)
                    ctrl.append(geo_ctrl)
                if args.controller == 'lqr':
                    lqr_ctrl = LQRController(env, self.linear_models[i])
                    ctrl.append(lqr_ctrl)
                if args.controller == 'dlqr':
                    dLQR = DecentralizedLQR(env, self.linear_models[i])
                    dLQR.K = computed_K
                    ctrl.append(dLQR)

        # Run the simulation
        START = time.time()
        action = np.zeros((args.num_drones, 4))
        obs, _, _, _, _ = env.step(action)
        CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)
        t = 0
        for i in range(CTRL_STEPS):
            for j in range(args.num_drones):
                if do_lemniscate:
                    pos, vel, acc, yaw, omega = traj(t)
                    ctrl[j].set_desired_trajectory(j, desired_pos=pos, desired_vel=vel, desired_acc=acc, desired_yaw=yaw,
                                                   desired_omega=omega)
                else:
                    ctrl[j].set_desired_trajectory(j, desired_pos=self.TARGET_POSITIONS[j, :], desired_vel=np.zeros((3,)),
                                                   desired_acc=np.zeros((3,)), desired_yaw=self.TARGET_RPYS[j, :][2],
                                                   desired_omega=0)

                action[j, :], u = ctrl[j].compute(obs[j])

                # logging.info(f"Action:\n {action}")
                # logging.info(f"Obs:\n {obs[0]}")
                # logging.info(f"Target Pos:\n {self.TARGET_POSITIONS[0, :]}")
                # logging.info(f"Target RPY:\n {self.TARGET_RPYS[0, :]}")

            # Apply the control input
            obs, _, _, _, _ = env.step(action)
            self.obs = obs
            if t >= 1:
                print("t>=1")
            self.observations.append(obs)
            self.obs_ts.append(t)
            t += env.CTRL_TIMESTEP
            env.render()
            sync(i, START, env.CTRL_TIMESTEP)
        # Close the environment
        env.close()

    def geometric_xdot(self, obs):
        # obs: [x, y, z, q0, q1, q2, q3, vx, vy, vz, wx, wy, wz]
        # action: [rpm1, rpm2, rpm3, rpm4]
        # return: xdot: [vx, vy, vz, wx, wy, wz, ax, ay, az, alpha_x, alpha_y, alpha_z]
        obs = np.array(obs)
        action = obs[16:20]  # clipped action
        action = np.array(action)
        u = conversions.action_to_input(self.env, action)
        a = np.zeros((3,))
        a[2] = u[0] / self.env.M  # z acceleration, to be rotated to world frame
        R = Rotation.from_quat(obs[3:7]).as_matrix()
        v_world = obs[10:13]
        w_bodyframe = np.matmul(R.T, obs[13:16])
        x_dot = np.zeros((12,))
        x_dot[0:3] = v_world
        x_dot[3:6] = w_bodyframe
        x_dot[6:9] = R.T @ a
        # x_dot[9:] = angular_acceleration TODO populate if desired

        return x_dot

    def circle_initialize(self):
        args = self.args
        # initialize the drones in a circle with radius args.init_rad centered around the first drone at (0,0,0)
        self.INIT_XYZS = np.zeros((args.num_drones, 3))

        for i in range(1, args.num_drones):  # first drone starts at (0,0,0) so don't initialize on circle
            self.INIT_XYZS[i, 0] = args.init_rad * np.cos((i / args.num_drones) * 2 * np.pi)
            self.INIT_XYZS[i, 1] = args.init_rad * np.sin((i / args.num_drones) * 2 * np.pi)
            self.INIT_XYZS[i, 2] = 0.0

        # initialize the target positions and orientations to just be a vertical offset of starting positions

        for i in range(args.num_drones):
            self.INIT_RPYS[i, 2] = 0
            self.TARGET_POSITIONS[i, 0] = self.INIT_XYZS[i, 0]
            self.TARGET_POSITIONS[i, 1] = self.INIT_XYZS[i, 1]
            self.TARGET_POSITIONS[i, 2] = self.INIT_XYZS[i, 2] + self.starting_target_offset

        for i in range(args.num_drones):
            self.TARGET_RPYS[i, 0] = 0
            self.TARGET_RPYS[i, 1] = 0
            self.TARGET_RPYS[i, 2] = np.pi / 2


if __name__ == "__main__":
    ARGS = parse_args()
    geo = GeometricEnv(ARGS, circle_init=True)
    env = geo.create_env()
    # computed_K, theta = geo.warm_up_only()
    geo.fedCE()
    # geo.do_control(do_lemniscate=False)
    exit()
    geo.args.controller = 'dlqr'
    env = geo.create_env()
    geo.do_control(do_lemniscate=True)
    thetaA = theta[:12, :].T
    thetaB = theta[12:, :].T
    with np.printoptions(precision=3, suppress=True):
        print("Theta A:\n ", thetaA)
        print("Theta B:\n ", thetaB)
    np.save("_observations.npy", geo.observations)
