from __future__ import division

import numpy as np
from cvxopt import matrix, solvers
from scipy import signal

# solvers.options['solver'] = 'mosek'
solvers.options['show_progress'] = False
from collections import namedtuple

"""
Helper class to compute CBF inequality constraint, where G*x <= h, where x=[u^T, slack]^T.
Note that for higher relative degree system, ECBF is used for safety.
This works for multi-agent collision avoidance.
"""


def all_unique(x):
    return len(x) == len(set(x))


# Define tuple to hold state bounds and define default values to None
fields = ('min_bounds', 'max_bounds')
StateBound = namedtuple('StateBound', fields)


class CBF:
    """
    Implement ECBF for both MAV & UGV based on "Safe Certificate-Based Maneuvers for Teams of Quadrotors Using Differential Flatness".
    We use this for ground vehicles (unicycles) because they are also differentially flat.
    Paper link: https://arxiv.org/pdf/1702.01075.pdf

        Solve QP in the form of
            min_v 0.5 v^T P v + q^T x
              s.t   Gv <= h (ECBF constraint & actuation constraint & room bounds)
                    Ax  = b (not used)
    """

    # TODO: related max vel, acc, jrk to max roll/pitch

    def __init__(self, xy_only, zscale=2.0, order=3, umax=None, safety_radius=1.0,
                 cbf_poles=np.array([-2.2, -2.4, -2.6]),
                 room_bounds=np.array([-4.25, 4.5, -3.5, 4.25, 1.0, 2.0]),
                 vmax=2,
                 amax=3,
                 jmax=4,
                 A=None,
                 B=None,
                 num_agents=1):
        self.xdim = A.shape[0] // num_agents  # state dimension
        self.zscale = zscale  # scale in z safety distance in super-ellipsoid
        self.order = order  # relative degree of output position w.r.t control input (jerk input: 3; snap input: 4)
        self.umax = umax  # maximum input magnitude
        self.safety_radius = safety_radius  # safety distance
        self.cbf_poles = cbf_poles  # the closer to 0 the poles are, the more conservative the cbf constraint
        self.xy_only = xy_only  # Whether ignore all z components for ground vehicles
        self.dim = 3  # !!Only handles 3D data
        self.num_agents = num_agents
        self.xdes = np.zeros((num_agents, self.xdim))
        # Set methods to get the {order}-th derivative of the barrier function for inter-agent collision avoidance

        self._hdots = self.custom_hdots
        self._ctrl_affine = self.custom_control_affine_terms
        self.state_bounds = [StateBound(None, None) for _ in range(self.order)]
        if room_bounds is not None:
            assert len(
                room_bounds) == 6, "Require xmin, xmax, ymin, ymax, zmin, zmax (6 values), but len(room_bounds)={}".format(
                len(room_bounds))
            xmin, xmax, ymin, ymax, zmin, zmax = room_bounds
            self.min_room_bounds = np.array([xmin, ymin, zmin])
            self.max_room_bounds = np.array([xmax, ymax, zmax])
            assert (
                    self.min_room_bounds < self.max_room_bounds).all(), "Not all min_room_bounds values are less than max_room_bounds. Current room_bounds={}".format(
                room_bounds)
            assert self.order >= 1 and self.order <= 4, "Only support relative degree between 1 (vel-controlled) and 4 (snp-controlled) system."
            assert len(cbf_poles) == self.order, "Number of specified CBF poles ({})does not match order ({})".format(
                len(self.cbf_poles), self.order)

            # Also get the bounds for vel, acc, jrk, TODO: load and consolidate with _build_box_const function
            self.vmax = vmax
            self.amax = amax
            self.jmax = jmax

            for i in range(self.order):
                if i == 0:  # Position bounds
                    self.state_bounds[i] = StateBound(min_bounds=np.array([xmin, ymin, zmin]),
                                                      max_bounds=np.array([xmax, ymax, zmax]))
                elif (i == 1) and (self.vmax is not None):  # Velocity bounds
                    self.state_bounds[i] = StateBound(min_bounds=np.array([-self.vmax] * 3),
                                                      max_bounds=np.array([self.vmax] * 3))
                elif (i == 2) and (self.amax is not None):  # Acceleration bounds
                    self.state_bounds[i] = StateBound(min_bounds=np.array([-self.amax] * 3),
                                                      max_bounds=np.array([self.amax] * 3))
                elif (i == 3) and (self.jmax is not None):  # Jerk bounds
                    self.state_bounds[i] = StateBound(min_bounds=np.array([-self.jmax] * 3),
                                                      max_bounds=np.array([self.jmax] * 3))

        self.A = A
        self.B = B

        # Generate feedback gain for the integrator system
        self.F = np.zeros((self.order, self.order))
        for i in range(self.order - 1):
            self.F[i, i + 1] = 1.0
        self.G = np.zeros((self.order, 1))
        self.G[-1, 0] = 1.0
        self.Kcbf = np.asarray(signal.place_poles(self.F, self.G, cbf_poles).gain_matrix)

    def set_xdes(self, xdes):
        if xdes.shape != self.xdes.shape:
            if xdes.shape[0] == self.xdes.shape[0] * self.xdes.shape[1]:
                self.xdes = xdes.reshape(self.xdes.T.shape).T  # TODO ensure the reshaping works properly
            else:
                raise ValueError("xdes shape {} does not match expected shape {}".format(xdes.shape, self.xdes.shape))
        self.xdes = xdes

    def rectify(self, x, uhat, ignore_zmin=False, x_obs=None, obs_r_list=None):
        # x is a list of (order, 3) np array; uhat is a list of (3, ) np array
        N = len(uhat)
        P = 2 * np.eye(4 * N)
        q = -2 * np.hstack(uhat).reshape(4 * N, )
        assert q.shape == (4 * N,), q.shape

        G, h = self._build_ineq_const(x, ignore_pos_zmin=ignore_zmin, x_obs=x_obs, obs_r_list=obs_r_list)

        # Solve for minimally invasive control (if possible)
        try:
            sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))  # without equality constraint
            sol_x = sol['x']
            u = [np.asarray(sol_x[3 * i:3 * (i + 1)]).reshape((3,)) for i in range(N)]
            return True, u
        except Exception as e:
            # rospy.logerr("MAVCBF cannot rectify. Error: {}".format(str(e)))
            return False, None

    def get_m12(self):
        return 1 / self.B[5, 0], 1 / self.B[5, 0]  # TODO ensure that we get the correct masses

    def get_g12(self):
        return self.A[3, 1], self.A[3, 1]

    # def custom_hdots(self, xi, xj, safety_dist, xi_des, xj_des, i=0, j=1):
    #     if self.order == 2:
    #         return self.custom_hdots_second(xi, xj, safety_dist, xi_des, xj_des, i, j)
    #     elif self.order == 3:
    #         return self.custom_hdots_3rd(xi, xj, safety_dist, xi_des, xj_des, i, j)
    #     else:
    #         raise ValueError("Only support relative degree between 2 (thrust) and 3 (yank) system")

    def custom_hdots(self, xi, xj, safety_dist, xi_des, xj_des, i=0, j=1):
        # Get 0-th to {order-1}-th time derivatives for the ECBF constraint given "ellipsoid" super-ellipsoid
        hdots = [None] * (self.order)
        A, B = self.getABij(i, j)
        c = self.zscale
        xhat = np.hstack((xi, xj)) - np.hstack((xi_des, xj_des))

        ex, ey, ez = (xi - xj)[-3:]
        for i in range(self.order):
            if i == 0:
                hdots[i] = (ex ** 2 + ey ** 2) ** 2 + (ez / c) ** 4 - safety_dist ** 4
            elif i == 1:
                dhde = np.zeros(self.xdim)
                dhde[-3] = 4 * ex * (ex ** 2 + ey ** 2)
                dhde[-2] = 4 * ey * (ex ** 2 + ey ** 2)
                dhde[-1] = 4 * ez ** 3 / c ** 4
                dhdx = np.zeros((1, 2 * self.xdim))
                dhdx[0, :self.xdim] = dhde
                dhdx[0, self.xdim:] = -dhde
                hdots[i] = (dhdx @ A @ xhat)[0]  # result should be 1x1, so get the value
            elif i == 2:
                dhde = np.zeros(self.xdim)

                dhde[6] = 4 * ex * (ex ** 2 + ey ** 2)
                dhde[7] = 4 * ey * (ex ** 2 + ey ** 2)
                dhde[8] = 4 * ez ** 3 / c ** 4
                dhdx = np.zeros((1, 2 * self.xdim))
                dhdx[0, :self.xdim] = dhde
                dhdx[0, self.xdim:] = -dhde
                d2hde2 = np.zeros((self.xdim, self.xdim))
                d2hde2[6, 6] = 12 * ex ** 2 + 4 * ey ** 2
                d2hde2[6, 7] = 8 * ex * ey
                d2hde2[7, 6] = 8 * ex * ey
                d2hde2[7, 7] = 4 * ex ** 2 + 12 * ey ** 2
                d2hde2[8, 8] = 12 * ez ** 2 / c ** 4
                d2hdx2 = np.zeros((2 * self.xdim, 2 * self.xdim))
                d2hdx2[:self.xdim, :self.xdim] = d2hde2
                d2hdx2[self.xdim:, self.xdim:] = d2hde2
                d2hdx2[:self.xdim, self.xdim:] = -d2hde2
                d2hdx2[self.xdim:, :self.xdim] = -d2hde2
                Axhat = A @ xhat
                Lfh = dhdx @ A @ Axhat + Axhat.T @ d2hdx2 @ Axhat
                hdots[i] = Lfh[0]
        return hdots

    def getABij(self, i, j):
        Ai = self.A[self.xdim * i:self.xdim * (i + 1), self.xdim * i:self.xdim * (i + 1)]
        Aj = self.A[self.xdim * j:self.xdim * (j + 1), self.xdim * j:self.xdim * (j + 1)]
        Bi = self.B[self.xdim * i:self.xdim * (i + 1), 4 * i:4 * (i + 1)]
        Bj = self.B[self.xdim * j:self.xdim * (j + 1), 4 * j:4 * (j + 1)]
        A = np.zeros((2 * self.xdim, 2 * self.xdim))
        B = np.zeros((2 * self.xdim, 8))
        A[:self.xdim, :self.xdim] = Ai
        A[self.xdim:, self.xdim:] = Aj
        B[:self.xdim, :4] = Bi
        B[self.xdim:, 4:] = Bj

        return A, B

    def custom_control_affine_terms(self, xi, xj, xi_des, xj_des, i=0, j=1):
        # Get control affine terms for the ECBF constraint given "circular" super-ellipsoid
        # L^{order}f and LgL^{order-1}f terms

        # TODO we can save compute by only computing the LgLfh for the first input, and the second is always the negative
        Lfh = 0
        LgLfh = np.zeros((4,))
        A, B = self.getABij(i, j)
        c = self.zscale
        xhat = np.hstack((xi, xj)) - np.hstack((xi_des, xj_des))

        ex, ey, ez = (xi - xj)[-3:]
        if self.order == 2:
            dhde = np.zeros(self.xdim)

            dhde[-3] = 4 * ex * (ex ** 2 + ey ** 2)
            dhde[-2] = 4 * ey * (ex ** 2 + ey ** 2)
            dhde[-1] = 4 * ez ** 3 / c ** 4
            dhdx = np.zeros((1, 2 * self.xdim))
            dhdx[0, :self.xdim] = dhde
            dhdx[0, self.xdim:] = -dhde
            d2hde2 = np.zeros((self.xdim, self.xdim))
            d2hde2[-3, -3] = 12 * ex ** 2 + 4 * ey ** 2
            d2hde2[-3, -2] = 8 * ex * ey
            d2hde2[-2, -3] = 8 * ex * ey
            d2hde2[-2, -2] = 4 * ex ** 2 + 12 * ey ** 2
            d2hde2[-1, -1] = 12 * ez ** 2 / c ** 4
            d2hdx2 = np.zeros((2 * self.xdim, 2 * self.xdim))
            d2hdx2[:self.xdim, :self.xdim] = d2hde2
            d2hdx2[self.xdim:, self.xdim:] = d2hde2
            d2hdx2[:self.xdim, self.xdim:] = -d2hde2
            d2hdx2[self.xdim:, :self.xdim] = -d2hde2

            Axhat = A @ xhat
            Lfh = dhdx @ A @ Axhat + Axhat.T @ d2hdx2 @ Axhat

            LgLfh = dhdx @ A @ B + Axhat.T @ d2hdx2 @ B
        elif self.order == 3:
            dhde = np.zeros(self.xdim)

            dhde[-3] = 4 * ex * (ex ** 2 + ey ** 2)
            dhde[-2] = 4 * ey * (ex ** 2 + ey ** 2)
            dhde[-1] = 4 * ez ** 3 / c ** 4
            dhdx = np.zeros((1, 2 * self.xdim))
            dhdx[0, :self.xdim] = dhde
            dhdx[0, self.xdim:] = -dhde
            d2hde2 = np.zeros((self.xdim, self.xdim))
            d2hde2[-3, -3] = 12 * ex ** 2 + 4 * ey ** 2
            d2hde2[-3, -2] = 8 * ex * ey
            d2hde2[-2, -3] = 8 * ex * ey
            d2hde2[-2, -2] = 4 * ex ** 2 + 12 * ey ** 2
            d2hde2[-1, -1] = 12 * ez ** 2 / c ** 4
            d2hdx2 = np.zeros((2 * self.xdim, 2 * self.xdim))
            d2hdx2[:self.xdim, :self.xdim] = d2hde2
            d2hdx2[self.xdim:, self.xdim:] = d2hde2
            d2hdx2[:self.xdim, self.xdim:] = -d2hde2
            d2hdx2[self.xdim:, :self.xdim] = -d2hde2

            d3hde3 = np.zeros((self.xdim, self.xdim, self.xdim))
            #im writing this with, depth, row, column
            d3hde3[-3, -3, -3] = 24 * ex
            d3hde3[-3, -3, -2] = 8 * ey
            d3hde3[-3, -2, -3] = 8 * ey
            d3hde3[-3, -2, -2] = 8 * ex

            d3hde3[-2, -2, -2] = 24 * ey
            d3hde3[-2, -3, -3] = 8 * ey
            d3hde3[-2, -3, -2] = 8 * ex
            d3hde3[-2, -2, -3] = 8 * ex
            d3hde3[-1, -1, -1] = 24 * ez / c ** 4

            d3hdx3 = np.zeros((2 * self.xdim, 2 * self.xdim, 2*self.xdim))
            d3hdx3[:self.xdim, :self.xdim, :self.xdim] = d3hde3
            d3hdx3[:self.xdim, self.xdim:, self.xdim:] = d3hde3
            d3hdx3[:self.xdim, self.xdim:, :self.xdim] = -d3hde3
            d3hdx3[:self.xdim, :self.xdim, self.xdim:] = -d3hde3

            d3hdx3[self.xdim:, self.xdim:, self.xdim:] = -d3hde3
            d3hdx3[self.xdim:, :self.xdim, :self.xdim] = -d3hde3
            d3hdx3[self.xdim:, self.xdim:, :self.xdim] = d3hde3
            d3hdx3[self.xdim:, :self.xdim, self.xdim:] = d3hde3

            Axhat = A @ xhat
            AA = A @ A
            ddxL2h = dhdx @ AA + (AA @ xhat).T @ d2hdx2 + Axhat.T @ (
                        d2hdx2 @ A + Axhat.T @ d3hdx3) + (d2hdx2 @ Axhat).T @ A
            Lfh = ddxL2h @ Axhat
            LgLfh = ddxL2h @ B

        return Lfh, LgLfh

    def _build_interagent_const_ij(self, x, i, j):
        """Build Gij and hij for the ECBF constraint"""
        N = self.num_agents
        xi, xj = x[i], x[j]

        # Compute time derivatives of barrier functions (0~order-1)
        Ds = 2 * self.safety_radius
        # Compute affine term in control input

        Lfh, LgLfh = self._ctrl_affine(xi=xi, xj=xj, xi_des=self.xdes[i], xj_des=self.xdes[j], i=i, j=j)
        hdots = self._hdots(xi, xj, safety_dist=Ds, xi_des=self.xdes[i], xj_des=self.xdes[j], i=i, j=j)

        # Build ij constraint
        Gij = np.zeros(4 * N, )
        Gij[4 * i:4 * (i + 1)] = -LgLfh[:, :4]
        Gij[4 * j:4 * (j + 1)] = LgLfh[:, :4]
        hij = np.dot(self.Kcbf, hdots) + Lfh

        return Gij, hij

    def _build_box_const(self, x, i, ignore_zmin=False, derivative_order=0):
        """Box constraint for agent i"""
        N = self.num_agents

        # Check if state bound exists TODO: clean up, ensure that both min & max bounds should exist in initialization
        if (self.state_bounds[derivative_order].min_bounds is None) or (
                self.state_bounds[derivative_order].max_bounds is None):
            return np.zeros((0, 4 * N)), np.zeros((0))

        xi = x[i]
        if ignore_zmin:
            Gi = np.zeros((5, 4 * N))
            hi = np.zeros((5,))
        else:
            Gi = np.zeros((6, 4 * N))
            hi = np.zeros((6,))

        # Modify default cbf poles based on the derivative ordre of box constraint (e.g., vel box constraint does not depend on position)
        Kcbf_local = self.Kcbf.copy()
        Kcbf_local[0, :derivative_order] = 0

        # Upper bounds
        Gi[0:4, 4 * i:4 * i + 4] = np.eye(4)
        xi_max = np.zeros((self.order, 3))
        xi_max[derivative_order] = self.state_bounds[derivative_order].max_bounds  # xi_max[0] = self.max_room_bounds
        hi[:3] = np.dot(Kcbf_local, xi_max - xi)

        # Lower bounds
        Gi[4, 4 * i] = -1
        Gi[4, 4 * i + 1] = -1
        if not ignore_zmin:
            Gi[5, 4 * i + 2] = -1

        xi_min = np.zeros((self.order, 3))
        xi_min[derivative_order] = self.state_bounds[derivative_order].min_bounds  # xi_min[0] = self.min_room_bounds
        if not ignore_zmin:
            hi[3:6] = np.dot(Kcbf_local, xi - xi_min)
        else:
            hi[3:5] = np.dot(Kcbf_local, xi[:, :-1] - xi_min[:, :-1])

        return Gi, hi

    """Helpers for building inequality constraints for QP"""

    def _build_ineq_const(self, x, ignore_pos_zmin, x_obs, obs_r_list):
        if len(x.shape) == 1:  # stacked x, reshape
            x = x.reshape(self.xdes.T.shape).T

        # --- Some initial checks ---
        # Set common z if xy-only
        if self.xy_only:
            for xi in x:
                # xi[:,2] = self.min_room_bounds[2] # Set a common z
                xi[:, 2] = self.state_bounds[0].min_bounds[2]  # Set a common z

        # Are there non-controlled obstacles?
        if (x_obs is not None) and (obs_r_list is not None):
            # Process obstacles position list
            assert len(x_obs) == len(
                obs_r_list), "The lists for Obstacle positions and radii must have the same length. Right now {} & {}".format(
                len(x_obs), len(obs_r_list))
            if len(x_obs) > 0:
                assert x_obs[0].shape == (
                    self.order, 3), "Each obstacle state must match robot state of shape (order={}, 3))".format(
                    self.order)

            if self.xy_only:  # Edit obstacle's position such that only xy positions matter
                for x_obs_j in x_obs:
                    # x_obs_j[:,2] = self.min_room_bounds[2] # Set a common z as the robots. This should match ground robot's z too
                    x_obs_j[:, 2] = self.state_bounds[0].min_bounds[
                        2]  # Set a common z as the robots. This should match ground robot's z too
        # ---------------------------

        N = self.num_agents
        G = np.zeros((0, 4 * N))  # place holder
        h = np.zeros((0,))  # place holder

        # inter-agent collision avoidance
        for i in range(N - 1):
            for j in range(i + 1, N):
                Gij, hij = self._build_interagent_const_ij(x, i, j)
                G = np.vstack((G, Gij))
                h = np.hstack((h, hij))

        # max infinity norm of control input constraint
        if self.umax is not None:
            G_umax, h_umax = self._build_umax_const(N)
            G = np.vstack((G, G_umax))
            h = np.hstack((h, h_umax))

        # box constraint for states for all agents
        G_statebnd, h_statebnd = self._build_state_bound_const(x, ignore_pos_zmin=ignore_pos_zmin)
        G = np.vstack((G, G_statebnd))
        h = np.hstack((h, h_statebnd))

        # Avoiding other non-controlled static/dynamic obstacles
        if (x_obs is not None) and (obs_r_list is not None):
            G_obs, h_obs = self.custom_build_obstacles_const(x, x_obs, obs_r_list)
            # G_obs, h_obs = self._build_obstacles_const(x, x_obs, obs_r_list)
            G = np.vstack((G, G_obs))
            h = np.hstack((h, h_obs))

        return G, h

    def _build_obstacles_const(self, x, x_obs, obs_r_list):
        # TODO: check during drone landing. The ground robots should avoid the drones
        # TODO: verify that it works for moving obstacles
        # Collision avoidance of every agent with a list of obstacles (static or dynamic) and their radii

        N = len(x)
        N_obs = len(x_obs)
        N_const = N_obs * N

        G = np.zeros((N_const, 4 * N))
        h = np.zeros((N_const))

        for i, xi in enumerate(x):
            for j, (xj, obs_r) in enumerate(zip(x_obs, obs_r_list)):
                # Difference in pos, vel, acc, jrk, where z axis is scaled.
                xdots = [(xi[k] - xj[k]) * np.array([1.0, 1.0, 1.0 / self.zscale]) for k in range(self.order)]
                # Compute time derivatives of barrier functions (0~order-1)
                hdots = self._hdots(xdots, self.safety_radius + obs_r)
                # Compute affine term in control input
                Lfh, LgLfh = self._ctrl_affine(xdots)

                # Build ij constraint
                G[i * N_obs + j, 3 * i:3 * (i + 1)] = -LgLfh

                h[i * N_obs + j] = np.dot(self.Kcbf,
                                          hdots) + Lfh  # No additional term since obstacles are not controlled at the {order}-th derivative (e.g. order=4 for snap control)

        return G, h

    def custom_build_obstacles_const(self, x, x_obs, obs_r_list):
        N = self.num_agents

        N_obs = len(x_obs)
        N_const = N_obs * N

        G = np.zeros((N_const, 4 * N))
        h = np.zeros((N_const))

        for i, xi in enumerate(x):
            for j, (xj, obs_r) in enumerate(zip(x_obs, obs_r_list)):
                Ds_obs = (self.safety_radius + obs_r)
                xj_obs = np.zeros(self.xdim)
                xj_obs[-3:] = xj[
                    0]  # position of obstacle is the only part of its state. TODO understand the shape of x_obs and remove this [0]
                # we pass xj_des to be the same as xj so that Aj @ (xj-xj_des) = Aj @ 0 = 0 for all Aj to represent there is
                # no motion of the obstacle.
                Lfh, LgLfh = self._ctrl_affine(xi=xi, xj=xj_obs, xi_des=self.xdes[i], xj_des=xj_obs, i=i, j=j)
                # here the LgLfh is computed for both inputs stacked, we only need it for the first so we can drop the rest
                # TODO see custom_control_affine_terms for details on how to speed up

                hdots = self._hdots(xi, xj_obs, safety_dist=Ds_obs, xi_des=self.xdes[i], xj_des=xj_obs, i=i, j=j)
                # Build ij constraint
                G[i * N_obs + j, 4 * i:4 * (i + 1)] = -LgLfh[:, :4]
                h[i * N_obs + j] = np.dot(self.Kcbf, hdots) + Lfh
                # No additional term since obstacles are not controlled at the {order}-th derivative (e.g. order=4 for snap control)

        return G, h

    def _build_umax_const(self, N):
        # max infinity norm of control input constraint
        if len(np.array(self.umax)) > 1:
            G = np.vstack((np.eye(4 * N), -np.eye(4 * N)))
            h = np.zeros(2 * 4 * N)
            for i in range(2 * N):
                h[4 * i:4 * (i + 1)] = np.array(self.umax)
            # tile umax constraint.
        else:
            G = np.vstack((np.eye(4 * N), -np.eye(4 * N)))
            h = self.umax * np.ones(2 * 4 * N)

        return G, h

    def _build_state_bound_const(self, x, ignore_pos_zmin):
        # box constraint for states (including room bounds) for each agent i
        N = len(x)
        G = np.zeros((0, 4 * N))  # place holder
        h = np.zeros((0,))  # place holder
        for i in range(N):
            for j in range(self.order):
                Gi, hi = self._build_box_const(x, i, ignore_zmin=(ignore_pos_zmin and j == 0), derivative_order=j)
                G = np.vstack((G, Gi))
                h = np.hstack((h, hi))

        return G, h

    def update_cbf_gain(self, cbf_poles):
        """Dynamic reconfiguration helpers"""
        success = False
        try:
            assert len(cbf_poles) == self.order, "[MAVCBF] Number of cbf poles do not match order."
            assert all_unique(cbf_poles), "[MAVCBF] All cbf poles must be unique"
            assert (cbf_poles < 0).all(), "[MAVCBF] All cbf poles must be negative"
            self.cbf_poles = cbf_poles
            self.Kcbf = np.asarray(signal.place_poles(self.A, self.B, cbf_poles).gain_matrix)
            success = True
            # rospy.loginfo("[MAVCBF] Updated cbf poles to be {}".format(self.cbf_poles))
        except Exception as e:
            # rospy.logerr("Failed to set cbf_poles with error message: {}".format(str(e)))
            pass

        return success

    def update_zscale(self, zscale):
        success = False
        try:
            assert zscale > 0, "[MAVCBF] zscale must be positive"
            self.zscale = zscale
            success = True
            # rospy.loginfo("[MAVCBF] Updated zscale to be {}".format(self.zscale))
        except Exception as e:
            # rospy.logerr("Failed to set zscale with error message: {}".format(str(e)))
            pass
        return success

    def update_safety_radius(self, safety_radius):
        success = False
        try:
            assert safety_radius > 0, "[MAVCBF] safety_radius must be positive"
            self.safety_radius = safety_radius
            success = True
            # rospy.loginfo("[MAVCBF] Updated safety_dist to be {}".format(self.Ds))
        except Exception as e:
            # rospy.logerr("Failed to set safety_dist with error message: {}".format(str(e)))
            pass
        return success


class DroneCBF(CBF):
    """
    Wrapper Around CBF class where we can accept an environment + model and prepopulate most of the CBF args.
    """

    def __init__(self, env, lin_models,
                 zscale=2.0,
                 safety_radius=1,
                 cbf_poles=np.array([-2.2, -2.4]),
                 room_bounds=None,
                 vmax=None,
                 amax=None,
                 jmax=None,
                 omega_max=np.ones(3),
                 order=2
                 ):
        # For now we set all the bounds to zero because I believe we will need to change their indexing for the drone
        # state.
        self.num_agents = len(lin_models)
        self.xdim = lin_models[0].A.shape[0]
        A = np.zeros((self.xdim * self.num_agents, self.xdim * self.num_agents))
        B = np.zeros((self.xdim * self.num_agents, 4 * self.num_agents))
        for i, model in enumerate(lin_models):
            A[self.xdim * i:self.xdim * (i + 1), self.xdim * i:self.xdim * (i + 1)] = model.A
            B[self.xdim * i:self.xdim * (i + 1), 4 * i:4 * (i + 1)] = model.B
        super().__init__(xy_only=False, zscale=zscale, order=order,
                         umax=np.array([env.MAX_THRUST, omega_max[0], omega_max[1], omega_max[2]]),
                         safety_radius=safety_radius, cbf_poles=cbf_poles, room_bounds=room_bounds,
                         vmax=vmax, amax=amax, jmax=jmax, A=A, B=B, num_agents=self.num_agents)

        self.lin_models = lin_models
        self.env = env
