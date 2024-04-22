import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation


class QuadrotorDynamics:
    def __init__(self, sim_freq, init_position=None, init_rpys=None):
        # Hummingbird
        # Source: https://github.com/ethz-asl/rotors_simulator
        self.m = 6.77
        self.Jxx = 1.05
        self.Jyy = 1.05
        self.Jzz = 2.05
        self.g = 9.81
        self.sim_freq = int(sim_freq)
        self.dt = 1.0 / sim_freq

        # Noise
        self.mu = 0.0
        self.sigma_a = 3.0000e-03
        self.sigma_w = 1.9393e-05

        # Motor
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.max_thrust = 1.0
        self.max_torque = 0.05

        self.J = np.diag([self.Jxx, self.Jyy, self.Jzz])
        self.J_inv = np.linalg.inv(self.J)
        self.e3 = np.array([0.0, 0.0, 1.0])

        # Initial State
        if init_position is None:
            self.x = np.array([0.0, 0.0, 0.0])
        else:
            self.x = init_position
        if init_rpys is None:
            self.R = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape([3, 3])
        else:
            rotation = Rotation.from_euler('xyz', init_rpys)
            self.R = rotation.as_matrix()
        self.v = np.array([0.0, 0.0, 0.0])
        self.w = np.array([0.0, 0.0, 0.0])
        self.b_a = np.array([0.0, 0.0, 0.0])
        self.b_w = np.array([0.0, 0.0, 0.0])

    def hat_map(self, w):
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
        return w_hat

    def normalize_rotmat(self, unnormalized_rotmat):
        x_raw = unnormalized_rotmat[0:3]
        y_raw = unnormalized_rotmat[3:6]

        x = x_raw / np.linalg.norm(x_raw)
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z)
        y = np.cross(z, x)

        matrix = np.vstack((x, y, z))
        return matrix

    def getState(self):
        return np.hstack((self.x, self.R.flatten(), self.v, self.w, self.b_a, self.b_w))

    def dynamics(self, t, state, u):
        '''
        state: (18,)-shaped Numpy array consisting of (p, R, v, w)
               - linear velocity is represented in the world-frame
               - angular velocity is represented in the body-frame
        u    : (4,)-shaped Numpy array consisting of (thrust, torques)
        '''
        thrust, torques = u[0], u[1:4]
        x = state[0:3]
        R = state[3:12].reshape([3, 3])
        v = state[12:15]
        w = state[15:18]
        n_a = np.random.normal(self.mu, self.sigma_a, 3)
        n_w = np.random.normal(self.mu, self.sigma_w, 3)

        x_dot = v
        R_dot = np.matmul(R, self.hat_map(w))  # Abdullah: Need to be checked 'self.'
        v_dot = np.matmul(R, thrust * self.e3) / self.m - self.g * self.e3
        w_dot = np.matmul(self.J_inv, torques - np.cross(w, np.matmul(self.J, w)))
        b_a_dot = n_a
        b_w_dot = n_w
        y_dot = np.hstack((x_dot, R_dot.flatten(), v_dot, w_dot, b_a_dot, b_w_dot))
        return y_dot

    def step(self, action):
        y0 = self.getState()
        ivp = solve_ivp(fun=lambda t, y: self.dynamics(t, y, action), t_span=[0, self.dt], y0=y0)
        y = ivp.y[:, -1]
        self.x = y[0:3]
        self.R = self.normalize_rotmat(y[3:12])
        self.v = y[12:15]
        self.w = y[15:18]
        self.b_a = y[18:21]
        self.b_w = y[21:24]
        obs = self.getState()
        return obs

