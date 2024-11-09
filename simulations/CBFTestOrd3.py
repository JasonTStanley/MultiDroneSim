# code structure follows the style of Symplectic ODE-Net
# https://github.com/d-biswa/Symplectic-ODENet/blob/master/experiment-single-embed/data.py

import argparse
import os
import time
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from scipy.spatial.transform import Rotation
from obstacles import *
import utils
import utils.model_conversions as conversions
# Pybullet drone environment
from control import GeometricControl, LQRYankOmegaController, DecentralizedLQRYankOmega, YankOmegaController
from model import LinearizedYankOmegaModel
from trajectories import *
from cbf import DroneQPTracker, DroneCBF
from utils import obs_to_lin_model

print(os.environ['PYTHONPATH'])
exit()
DEFAULT_DRONES = DroneModel("cf2p")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True  # usually true
DEFAULT_PLOT = False  # usually true
DEFAULT_RECORD = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 100
DEFAULT_CONTROL_FREQ_HZ = 100
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NUM_DRONES = 7
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
    parser.add_argument('--init_rad', default=.2, type=float,
                        help='Initial radius of the drones (default: 1.0)', metavar='')
    parser.add_argument('--controller', default=controllers[0], type=str,
                        help=f'Controller to use from {controllers} (default: {controllers[0]})', metavar='')
    ARGS = parser.parse_args()
    return ARGS


class GeometricEnv:
    init_types = ['circle', 'lemniscate']
    def __init__(self, args, init_type='circle', lemniscate_a=1, center=np.array([0, 0, 0])):
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
        self.init_type = init_type
        self.lemniscate_a = lemniscate_a
        self.center = center

        if init_type == 'circle':
            self.starting_target_offset = 1  # initial goal is just 1m above start pose
            self.circle_initialize()
        elif init_type == 'lemniscate':
            self.starting_target_offset = 1
            self.lemniscate_initialize()


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
                         record=DEFAULT_RECORD,
                         user_debug_gui=args.user_debug_gui,
                         output_folder=args.output_folder
                         )
        # p.setPhysicsEngineParameter(numSubSteps=0)
        self.env = env

        self.linear_models = [LinearizedYankOmegaModel(env) for _ in range(args.num_drones)]

        r = self.env.KM / self.env.KF
        # convert between motor_thrusts and thrust/torques for a PLUS frame drone (CF2P) (is the Crazyflie 2.1 with plus config)
        self.conversion_mat = np.array([[1.0, 1.0, 1.0, 1.0],
                                        [0.0, self.env.L, 0.0, -self.env.L],
                                        [-self.env.L, 0.0, self.env.L, 0.0],
                                        [-r, r, -r, r]])
        return self.env

    def fedCE(self, num_iter=15):
        env = self.env
        args = self.args
        steps = 0
        dLQR = DecentralizedLQRYankOmega(env, self.linear_models)
        START = time.time()
        for n in range(num_iter):
            steps = self.fedCE_iteration(env, dLQR, START, steps, n, do_warmup=(n == 0), random_warmup=True)
            thetaA = dLQR.theta[:9 * args.num_drones, :].T
            thetaB = dLQR.theta[9 * args.num_drones:, :].T
            with np.printoptions(precision=3, suppress=True, linewidth=100000):
                print(f"n: {n}, steps: {steps}")
                print("Theta A:\n ", thetaA)
                print("Theta B:\n ", thetaB)

        env.close()
        return dLQR.K, dLQR.theta

    def fedCE_iteration(self, env, dLQR, START, steps, n, k=2, do_warmup=True, random_warmup=True, do_lemniscate=False,
                        do_print=False):
        Texp = n * k
        Tce = n * (k ** 3)
        Tw = 0
        if do_warmup:
            Tw = 25  # some small warmup

        args = self.args
        action = np.zeros((args.num_drones, 4))
        obs, _, _, _, _ = env.step(action)
        # CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)

        t = 0
        traj = Lemniscate(center=np.array([0, 0, .5]), omega=1, yaw_rate=.1)
        # warm up
        for i in range(Tw):
            phis = []
            e_tp1s = []
            x_des = np.zeros((9,))
            action = np.zeros((args.num_drones, 4))

            for j in range(args.num_drones):
                x = obs_to_lin_model(obs[j], dim=9)

                if random_warmup:
                    u = dLQR.sigma1()
                    act = dLQR.compute_low_level(u, obs[j], j)
                    # set x_des to be the starting point
                    x_des = np.zeros((9,))
                    x_des[0:3] = self.INIT_RPYS[j, :]
                    x_des[-3:] = self.INIT_XYZS[j, :]
                else:
                    pos, vel, acc, yaw, omega = traj(t)
                    x_des = np.hstack([[0, 0, yaw], vel, pos])
                    act, u = dLQR.LQR(obs[j], j, pos=pos, vel=vel, yaw=yaw, omega=omega)


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
                x_tp1 = obs_to_lin_model(obs[j], dim=9)
                x_des = np.zeros((9,))
                x_des[0:3] = self.INIT_RPYS[j, :]
                x_des[-3:] = self.INIT_XYZS[j, :]
                e_tp1 = dLQR.error_state(x_tp1, x_des)
                e_tp1s.append(e_tp1)

            # update the theta and V values
            if i != 0:  # skip first because the initial state of the drone can be far from the desired state
                dLQR.theta_update(phis, e_tp1s)

            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1

        t = 0
        # CE phase
        last_desired = np.zeros((args.num_drones, 9))
        dLQR.compute_controller()
        for i in range(Tce):
            # compute controller from current observations and use it to achieve some task
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
                    last_desired[j, :] = np.hstack([self.TARGET_RPYS[j, :], np.zeros((3,)),
                                                    self.TARGET_POSITIONS[j, :]])

            action, u = dLQR.compute(obs)

            obs, _, _, _, _ = env.step(action)
            t += env.CTRL_TIMESTEP
            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1

        # do exploration
        for i in range(Texp):
            phis = []
            e_tp1s = []
            action = np.zeros((args.num_drones, 4))
            for j in range(args.num_drones):
                x = obs_to_lin_model(obs[j], dim=9)
                x_des = last_desired[j, :]
                e = dLQR.error_state(x, x_des)

                u = dLQR.sigma_explore() #randomly choose a control input
                act = dLQR.compute_low_level(u, obs[j], j) # compute the action using low level controller

                # offset u by the equilibrium point
                u[0] = u[0] - env.M * env.G  # TODO replace with estimate?

                # since our model is formed on the error state, we must calculate e to learn theta
                action[j, :] = act
                phis.append(np.hstack([e, u]))  # use error state as input state to update phi

            obs, _, _, _, _ = env.step(action)

            for j in range(args.num_drones):
                # need to offset by the error
                x_des = last_desired[j, :]
                x_tp1 = obs_to_lin_model(obs[j], dim=9)
                e_tp1 = dLQR.error_state(x_tp1, x_des)
                e_tp1s.append(e_tp1)

            # update the theta and V values
            if i != 0:
                dLQR.theta_update(phis, e_tp1s)

            t += env.CTRL_TIMESTEP
            if do_print:
                env.render()
            sync(steps, START, env.CTRL_TIMESTEP)
            steps += 1
        return steps

    def do_control(self, trajs=None, qpTracker=None, render=False, computed_K=None, use_noisy_model=True, x_obs_list=None, obs_r_list=None):
        env = self.env
        args = self.args
        # for information about collecting a dataset using similar code, see https://github.com/altwaitan/DL4IO/blob/main/examples/pybullet/data_collection.py
        PYB_CLIENT = env.getPyBulletClient()
        PYB_CLIENT = env.getPyBulletClient()
        DRONE_IDS = env.getDroneIds()
        env._showDroneLocalAxes(0)

        # PID control for set point regulation
        ctrl = []
        if args.drone in [DroneModel.CF2X, DroneModel.CF2P]:
            if args.controller == 'dlqr':
                dLQR = DecentralizedLQRYankOmega(env, self.linear_models)
                dLQR.K = computed_K
                ctrl.append(dLQR)
            else:
                for i in range(args.num_drones):
                    if args.controller == "geometric":
                        geo_ctrl = GeometricControl(env)
                        ctrl.append(geo_ctrl)
                    if args.controller == 'lqr':
                        yo_controller = YankOmegaController(env)
                        lqr_ctrl = LQRYankOmegaController(env, self.linear_models[i], yo_controller, use_noisy_model=use_noisy_model)
                        ctrl.append(lqr_ctrl)

        # Run the simulation
        START = time.time()
        action = np.zeros((args.num_drones, 4))
        nominal_us = np.zeros((args.num_drones, 4))
        obs, _, _, _, _ = env.step(action)
        CTRL_STEPS = int(args.duration_sec * env.CTRL_FREQ)
        t = 0
        for i in range(CTRL_STEPS):
            for j in range(args.num_drones):
                if trajs is not None:
                    pos, vel, acc, yaw, omega = trajs[j](t)
                    controller = ctrl[0] if args.controller == 'dlqr' else ctrl[j]
                    controller.set_desired_trajectory(j, desired_pos=pos, desired_vel=vel, desired_acc=acc,
                                                      desired_yaw=yaw,
                                                      desired_omega=omega)
                else:

                    controller = ctrl[0] if args.controller == 'dlqr' else ctrl[j]
                    controller.set_desired_trajectory(j, desired_pos=self.TARGET_POSITIONS[j, :],
                                                      desired_vel=np.zeros((3,)),
                                                      desired_acc=np.zeros((3,)), desired_yaw=self.TARGET_RPYS[j, :][2],
                                                      desired_omega=0)

                if args.controller != 'dlqr':
                    action[j, :], u = ctrl[j].compute(obs[j], skip_low_level=qpTracker is not None)
                    nominal_us[j, :] = u
            # Apply the control input
            if args.controller == 'dlqr':
                action, u = ctrl[0].compute(obs, skip_low_level=qpTracker is not None)
                nominal_us = u


            # p.applyExternalForce(env.DRONE_IDS[0],
            #                      -1,  # -1 for the base, 0-3 for the motors
            #                      forceObj=[.001, 0, 0],  # a force vector
            #                      posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=PYB_CLIENT)

            if qpTracker is not None:
                #nominal_us stores the nominal control from whichever base controller is used, we must:
                # 1. offset these by hover force to give uhat
                # 2. modify the control input to satisfy the CBF and constraints using qpTracker
                # 3. add back the hover force
                # 4. use the low level controller to compute the action

                nominal_us[:, 0] = nominal_us[:, 0] - env.M * env.G
                xdes = np.zeros((args.num_drones, 10))
                for j in range(args.num_drones):
                    pos, vel, acc, yaw, omega = trajs[j](t)
                    xdes[j] = np.hstack([0, 0, yaw, self.env.G*self.env.M, vel, pos])

                u_safe = qpTracker.compute_control(obs, xdes, nominal_us, x_obs=x_obs_list, obs_r_list=obs_r_list)
                u_safe[:, 0] = u_safe[:, 0]
                for j in range(args.num_drones):
                    action[j, :] = ctrl[j].compute_low_level(u_safe[j, :], obs[j], j)

            obs, _, _, _, _ = env.step(action)
            self.obs = obs
            self.observations.append(obs)
            self.obs_ts.append(t)
            t += env.CTRL_TIMESTEP
            if render:
                print(f"Action: {action}")
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

    def lemniscate_initialize(self):
        args = self.args
        # initialize the drones in a circle with radius args.init_rad centered around the first drone at (0,0,0)
        self.INIT_XYZS = np.zeros((args.num_drones, 3))

        for i in range(0, args.num_drones):
            #create a lemniscate and get the 0 time position
            #make the phase shift some reasonable amount in the future,
            lem_i = Lemniscate(center=self.center, phase_shift=(2*np.pi / (args.num_drones + 0.25)) * i)
            pos, vel, acc, yaw, omega = lem_i(0)
            self.INIT_XYZS[i, 0] = pos[0]
            self.INIT_XYZS[i, 1] = pos[1]
            self.INIT_XYZS[i, 2] = pos[2]

        # initialize the target positions and orientations to just be a vertical offset of starting positions

        for i in range(args.num_drones):
            self.INIT_RPYS[i, 2] = 0
            self.TARGET_POSITIONS[i, 0] = self.INIT_XYZS[i, 0]
            self.TARGET_POSITIONS[i, 1] = self.INIT_XYZS[i, 1]
            self.TARGET_POSITIONS[i, 2] = self.INIT_XYZS[i, 2] + self.starting_target_offset

        for i in range(args.num_drones):
            self.TARGET_RPYS[i, 0] = 0
            self.TARGET_RPYS[i, 1] = 0
            self.TARGET_RPYS[i, 2] = 0

    def circle_initialize(self):
        args = self.args
        # initialize the drones in a circle with radius args.init_rad centered around the first drone at (0,0,0)
        self.INIT_XYZS = np.zeros((args.num_drones, 3))

        for i in range(1, args.num_drones):  # first drone starts at (0,0,0) so don't initialize on circle
            self.INIT_XYZS[i, 0] = args.init_rad * np.cos((i / args.num_drones) * 2 * np.pi) + self.center[0]
            self.INIT_XYZS[i, 1] = args.init_rad * np.sin((i / args.num_drones) * 2 * np.pi) + self.center[1]
            self.INIT_XYZS[i, 2] = self.center[2]

        # initialize the target positions and orientations to just be a vertical offset of starting positions

        for i in range(args.num_drones):
            self.INIT_RPYS[i, 2] = 0
            self.TARGET_POSITIONS[i, 0] = self.INIT_XYZS[i, 0]
            self.TARGET_POSITIONS[i, 1] = self.INIT_XYZS[i, 1]
            self.TARGET_POSITIONS[i, 2] = self.INIT_XYZS[i, 2] + self.starting_target_offset

        for i in range(args.num_drones):
            self.TARGET_RPYS[i, 0] = 0
            self.TARGET_RPYS[i, 1] = 0
            self.TARGET_RPYS[i, 2] = 0

def add_env_obstacles(env, x_obs_list, obs_r_list):
    for i in range(len(x_obs_list)):

        urdf_file = generate_sphere(obs_r_list[i])
        obs_center = x_obs_list[i][0]
        p.loadURDF(urdf_file, obs_center, useFixedBase=True)

if __name__ == "__main__":
    ARGS = parse_args()
    geo = GeometricEnv(ARGS, init_type='lemniscate', lemniscate_a=1)
    env = geo.create_env()
    # trajs = [WaitTrajectory(duration=20, position=geo.TARGET_POSITIONS[j]) for j in range(ARGS.num_drones)]
    trajs = [Lemniscate(a=1, center=np.array([0, 0 , 0.5]), omega=0.5, yaw_rate=0, phase_shift=(2*np.pi/(ARGS.num_drones + 0.25)) * num) for num in range(ARGS.num_drones)]
    # droneCBF = DroneCBF(env, geo.linear_models, safety_radius=0.25, zscale=1.5, order=3, cbf_poles=np.array([-2.2, -3.4, -5.6]))
    droneCBF = DroneCBF(env, geo.linear_models, safety_radius=0.125, zscale=2, order=3, cbf_poles=np.array([-3.0, -3.6, -5.6]))
    droneTracker = DroneQPTracker(droneCBF, num_robots=ARGS.num_drones, xdim=10, env=env, order=3)
    # x_obs_list = np.array([
    #     np.array([[0, 0, .5], np.zeros(3), np.zeros(3)]),
    # ])
    # obs_r_list = [.1, ]
    x_obs_list=None
    obs_r_list=None
    # add_env_obstacles(env, x_obs_list, obs_r_list)
    geo.do_control(trajs=trajs, qpTracker=droneTracker, render=True, computed_K=None, use_noisy_model=False,
                   x_obs_list=x_obs_list, obs_r_list=obs_r_list)
    exit()

    # # trajs = [Lemniscate(center=np.array([0, 0, .5]), omega=1.5, yaw_rate=0) for _ in range(ARGS.num_drones)]
    # geo.do_control(trajs=trajs, render=False)
    # np.save("observations_omega.npy", geo.observations)
    # exit()
    # computed_K, theta = geo.warm_up_only()
    computed_K, theta = geo.fedCE(num_iter=20)
    geo.args.controller = 'dlqr'
    env = geo.create_env()
    # traj = Lemniscate(center=np.array([0, 0, .5]), omega=0.5, yaw_rate=0.2)
    delta = np.array([0, 5, 0])
    delta2 = np.array([0, 5, 0])
    trajs = [CompoundTrajectory([LineTrajectory(start=geo.INIT_XYZS[idx], end=geo.TARGET_POSITIONS[idx], speed=.5),
                                 WaitTrajectory(duration=1, position=geo.TARGET_POSITIONS[idx]),
                                 LineTrajectory(start=geo.TARGET_POSITIONS[idx],
                                                end=geo.TARGET_POSITIONS[idx] + delta,
                                                speed=1),
                                 LineTrajectory(start=(geo.TARGET_POSITIONS[idx] + delta),
                                                end=(geo.TARGET_POSITIONS[idx]), speed=1)])
             for idx in range(ARGS.num_drones)]
    geo.do_control(trajs=trajs, computed_K=computed_K, render=False, use_noisy_model=False)
    np.save("observations2.npy", geo.observations)
