import matplotlib.pyplot as plt
import numpy as np

from trajectories.Lemniscate import Lemniscate


def plot_error_from_nominal(data, show=True):
    # data is in format pg_pred, ng_pred, Iv, Iw, Iinv, minv
    # ground truth is env.M = 0.027, env.J = np.diag([0.000024, 0.000024, 0.000032]) and env.G = 9.8

    # a simple plot first is maybe just the vector diff from the gt values to the results of FedCE
    diffs = [[], []]
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur = data[i][j]
            gt = np.hstack([9.8, -9.8, np.eye(3).flatten(), np.eye(3).flatten(),
                            np.linalg.inv(np.diag([0.000024, 0.000024, 0.000032])).flatten(), 1 / 0.027])

            diff = gt - cur
            diffs[j].append(diff)

    print(diffs[0][-1])
    print(diffs[1][-1])

    plt.plot(diffs[0])
    if show:
        plt.show()
    plt.plot(diffs[1])
    if show:
        plt.show()


def plot_matrix_norm(pred_thetas, show=True):
    fig = plt.figure()
    plt.title("Matrix Norm: $||\\theta^*_i - \hat{\\theta}_i||_2$")
    for i in range(pred_thetas.shape[0]):
        theta_hat = pred_thetas[i]
        A = np.zeros((12, 12))
        B = np.zeros((12, 4))
        A[0:3, 3:6] = np.eye(3)
        A[9:, 6:9] = np.eye(3)
        # self.A[6, 1] = -self.g
        # self.A[7, 0] = self.g
        g = 9.8
        Ixx = 2.3951e-5
        Iyy = 2.3951e-5
        Izz = 3.2347e-5
        mass = 0.027
        A[6, 1] = g
        A[7, 0] = -g

        B[8, 0] = 1.0 / mass
        B[3:6, 1:] = np.diag([1 / Ixx, 1 / Iyy, 1 / Izz])
        theta_star = np.hstack([A, B])
        theta_star_stacked = np.repeat(theta_star[np.newaxis, :, :], theta_hat.shape[0], axis=0)
        norms = np.linalg.norm(theta_hat - theta_star_stacked, axis=(1, 2))

        plt.plot(norms, label=f'Robot {i + 1}')
    plt.xlabel("Number of Updates")
    plt.legend()
    plt.savefig("plots/matrix_norm.png")
    if show:
        plt.show()


def plot_prediction_errors(pred_errs, show=True, save_path="plots/prediction_errors.png"):
    fig = plt.figure()
    plt.title("Prediction Errors: $|| \dot{x}_i - \dot{\hat{x}}_i||_2$")
    for i in range(len(pred_errs) // 2):
        plt.plot(pred_errs[i], label=f'$\hat{{\\theta}}_{i + 1}$ Pred')
    for i in range(len(pred_errs) // 2, len(pred_errs)):
        plt.plot(pred_errs[i], label=f'$\\theta_{i + 1}^*$ Pred')
    plt.xlabel("Number of Updates")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_tracking_error(bad_model, learned, wind=False, show=True):
    fig = plt.figure(figsize=(6, 4))
    num_robots = bad_model.shape[1]
    if wind:
        plt.title("Tracking Error with Wind: $|| x_i(t) - x_i^*(t)||_2$")
    else:
        plt.title("Tracking Error: $|| x_i(t) - x_i^*(t)||_2$")
    # for i in range(len(bad_model)):
    #     plt.plot(bad_model[i], label=f'Robot {i+1} Bad Model')
    trajs = [Lemniscate(center=np.array([0, 0, .5]), omega=1.5, yaw_rate=0.0, phase_shift=(-np.pi / 4) * (num - 1)) for
             num in range(num_robots)]
    dt = 1.0 / 100.0

    for i in range(num_robots-1):
        # pos_err only for now
        traj_points = np.array([np.hstack(trajs[i](dt * j)) for j in range(len(bad_model))])
        label_str = ['x', 'y']
        data_dict = {}
        for j in range(2):

            # plt.plot(learned[:, i, j], label=f'Robot {i + 1} Learned {label_str[j]}')
            # plt.plot(bad_model[:, i,j], label=f'Robot {i + 1} Incorrect Mass Model {label_str[j]}')
            # plt.plot(traj_points[:, j], label=f'Robot {i + 1} Desired {label_str[j]}')

            data_dict[f'Learned Model {label_str[j]}'] = learned[:, i, j] - traj_points[:, j]
            data_dict[f'Initial Model {label_str[j]}'] = bad_model[:, i, j] - traj_points[:, j]

        plt.boxplot(data_dict.values(), labels=data_dict.keys(), showfliers=False)

            # learned_mean = np.mean(learned[:, i, j] - traj_points[:, j])
            # learned_std = np.std(learned[:, i, j] - traj_points[:, j])
            # bad_mean = np.mean(bad_model[:, i, j] - traj_points[:, j])
            # bad_std = np.std(bad_model[:, i, j] - traj_points[:, j])
            # print(f"Robot {i + 1} Learned {label_str[j]} Mean: {learned_mean} Std: {learned_std}")
            # print(f"Robot {i + 1} Bad Model {label_str[j]} Mean: {bad_mean} Std: {bad_std}")


    plt.legend()
    plt.savefig("plots/tracking_error.png")
    if show:
        plt.show()


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True
    })

    pred_errs = np.load("simulations/pred_errors.npy", allow_pickle=True)
    plot_prediction_errors(pred_errs, show=False)
    pred_thetas = np.load("simulations/pred_thetas.npy", allow_pickle=True)
    plot_matrix_norm(pred_thetas, show=False)
    bad_model = np.load("simulations/observations_lem_bad_mass.npy", allow_pickle=True)
    learned = np.load("simulations/observations_lem_learned.npy", allow_pickle=True)
    plot_tracking_error(bad_model, learned, show=False)
    wind_bad_model = np.load("simulations/wind_observations_lem_bad_mass.npy", allow_pickle=True)
    wind_learned = np.load("simulations/wind_observations_lem_learned.npy", allow_pickle=True)
    plot_tracking_error(wind_bad_model, wind_learned, wind=True, show=True)
