import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from model.linearized import LinearizedModel
from EnvGeometric import GeometricEnv, parse_args
from model.dynamics import QuadrotorDynamics
import utils.model_conversions as conversions
#here we can graph a comparison of the geometric dynamics model and the linear model to show the differences between them.

def main():
    #define the geometric model
    args = parse_args()
    geometric = GeometricEnv(args, circle_init=True)
    geometric.TARGET_POSITIONS[0, :] = np.array([3, 3, 1.5])
    geometric.TARGET_RPYS[0, :] = np.array([0, 0, 0])
    # create env
    env = geometric.create_env(gui=False)
    #after the step control is done we can get the observations.
    try:
        observations = np.load("observations.npy")
        obs_ts = np.load("obs_ts.npy")
    except:
        geometric.do_control()
        observations = np.array(geometric.observations).squeeze()
        obs_ts = np.array(geometric.obs_ts).squeeze()
        np.save("observations.npy", observations)
        np.save("obs_ts.npy", obs_ts)

    #define the linear model
    linear = LinearizedModel(env)
    #get the x_dot from the linear model
    plot_values = 'vx,vy,vz,wx,wy,wz,ax,ay,az'.split(',')
    # plot_values = 'wx,wy,wz'.split(',')
    xdot_dict = {'vx': 0, 'vy': 1, 'vz': 2, 'wx': 3, 'wy': 4, 'wz': 5, 'ax': 6, 'ay': 7, 'az': 8, 'alpha_x': 9, 'alpha_y': 10, 'alpha_z': 11}
    linear_xdot_dict = {'wx': 0, 'wy': 1, 'wz': 2, 'alpha_x': 3, 'alpha_y': 4, 'alpha_z': 5, 'ax': 6, 'ay': 7, 'az': 8, 'vx': 9, 'vy': 10, 'vz': 11}

    linear_x_dict = {'roll': 0, 'pitch': 1, 'yaw': 2, 'wx': 3, 'wy': 4, 'wz': 5, 'vx': 6, 'vy': 7, 'vz': 8, 'x': 9, 'y': 10, 'z': 11}
    x_dot_linear = []
    x_dot_geometric = []
    x_lin_obs = []

    res = roll_out_linear_system(linear, observations, obs_ts)
    y = res.y
    t = res.t

    geo_dynamics = QuadrotorDynamics(env.PYB_FREQ)
    geo_dynamics.load_env_params(env)
    for i in range(observations.shape[0]):
        x_dot_linear.append(linear.calc_xdot_from_obs(observations[i]))
        # x_dot_geometric.append(geometric.geometric_xdot(observations[i]))
        #action to thrust / torques
        u = conversions.action_to_input(env, observations[i][16:])
        # print(u)
        x_dot_geometric.append(conversions.geo_x_dot_to_linear(geo_dynamics.dynamics(None, conversions.obs_to_geo_model(observations[i]), u)))
        x_lin_obs.append(conversions.obs_to_lin_model(observations[i]))

    x_dot_linear = np.array(x_dot_linear)
    x_dot_geometric = np.array(x_dot_geometric)
    x_lin_obs = np.array(x_lin_obs)

    #plot the results
    fig, axs = plt.subplots(len(plot_values))
    for i, value in enumerate(plot_values):
        axs[i].plot(x_dot_linear[:, linear_xdot_dict[value]], label="Linear")
        # axs[i].plot(x_dot_geometric[:, xdot_dict[value]], label="Geometric")
        axs[i].plot(x_dot_geometric[:, linear_xdot_dict[value]], label="Geometric")
        axs[i].set_title(value)
        if i == 0:
            axs[i].legend()
    plt.show()

    fig, axs = plt.subplots(len(linear_x_dict.keys()))
    for i, value in enumerate(linear_x_dict.keys()):
        axs[i].plot(t, y[linear_x_dict[value], :], label="Rolled out")
        axs[i].plot(obs_ts, x_lin_obs[:, linear_x_dict[value]], label="Geometric")
        axs[i].set_title(value)
        if i == 0:
            axs[i].legend()
    plt.show()


def roll_out_linear_system(linear, observations, obs_ts):
    from scipy.integrate import solve_ivp

    def f(t, x):
        #get the closest in the past observation to the time t
        time_diffs = np.abs(obs_ts - t)
        closest_idx = np.argmin(time_diffs)
        if obs_ts[closest_idx] > t:
            closest_idx -= 1
        obs = observations[closest_idx]
        return linear.calc_xdot(x,obs[16:])
    x0 = conversions.obs_to_lin_model(observations[0])
    res = solve_ivp(f, [0, obs_ts[-1]], x0, t_eval=obs_ts)
    return res

if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = [10, 10]
    main()
    print("done.")

