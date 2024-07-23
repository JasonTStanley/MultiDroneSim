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

    def __init__(self, xy_only, zscale=2.0, order=3, umax=1e4, safety_radius=1.0,
                 cbf_poles=np.array([-2.2, -2.4, -2.6]),
                 room_bounds=np.array([-4.25, 4.5, -3.5, 4.25, 1.0, 2.0]),
                 vmax=2,
                 amax=3,
                 jmax=4,
                 ellipsoid_shape='circular'):
        self.zscale = zscale  # scale in z safety distance in super-ellipsoid
        self.order = order  # relative degree of output position w.r.t control input (jerk input: 3; snap input: 4)
        self.umax = umax  # maximum input magnitude
        self.safety_radius = safety_radius  # safety distance
        self.cbf_poles = cbf_poles  # the closer to 0 the poles are, the more conservative the cbf constraint
        self.xy_only = xy_only  # Whether ignore all z components for ground vehicles
        self.dim = 3  # !!Only handles 3D data

        # Set methods to get the {order}-th derivative of the barrier function for inter-agent collision avoidance
        self.ellipsoid_shape = ellipsoid_shape
        if self.ellipsoid_shape == "circular":  # circular in xy
            self._hdots = self._hdots_circ_ellipsoid
            self._ctrl_affine = self._control_affine_terms_circ_ellipsoid
        elif self.ellipsoid_shape == "rectangular":  # rectangular-ish in xy
            self._hdots = self._hdots_rect_ellipsoid
            self._ctrl_affine = self._control_affine_terms_rect_ellipsoid

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

        self.state_bounds = [StateBound(None, None) for _ in range(self.order)]
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

        # Generate feedback gain for the integrator system
        self.A = np.zeros((self.order, self.order))
        for i in range(self.order - 1):
            self.A[i, i + 1] = 1.0
        self.B = np.zeros((self.order, 1))
        self.B[-1, 0] = 1.0
        self.Kcbf = np.asarray(signal.place_poles(self.A, self.B, cbf_poles).gain_matrix)

    def rectify(self, x, uhat, ignore_zmin=False, x_obs=None, obs_r_list=None):
        # x is a list of (order, 3) np array; uhat is a list of (3, ) np array
        N = len(uhat)
        P = 2 * np.eye(3 * N)
        q = -2 * np.hstack(uhat).reshape(3 * N, )
        assert q.shape == (3 * N,), q.shape

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

    def custom_hdots_crazyflie(self, xi,xj,xi_des, xj_des, ui,uj, safety_dist):
        '''The hdots for the 2nd order system having roll, pitch, yawrate, and thrust as input '''

    def custom_hdots_omega(self, xi, xj, xi_des, xj_des, ui, uj, safety_dist):
        '''The hdots for the 2nd/3rd order system having angular velocity as input A \in 9x9'''
        # Get 0-th to {order-1}-th time derivatives for the ECBF constraint given "circular" super-ellipsoid
        hdots = [None] * self.order
        A = self.A
        B = self.B
        c = self.zscale
        for i in range(self.order):
            if i == 0:
                x, y, z = (xi - xj)[-3:]  # last 3 is always position -> [xi-xj, yi-yj, (zi-zj)]
                hdots[i] = (x ** 2 + y ** 2) ** 2 + (z / c) ** 4 - safety_dist ** 4

            elif i == 1:
                [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17] = np.vstack((xi, xj))
                [xdes0, xdes1, xdes2, xdes3, xdes4, xdes5, xdes6, xdes7, xdes8, xdes9, xdes10, xdes11, xdes12, xdes13,
                 xdes14, xdes15, xdes16, xdes17] = np.vstack((xi_des, xj_des))

                hdots[i] = 4 * (x12 - xdes12) * (x15 - x6) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) + 4 * (
                        x13 - xdes13) * (
                                   x16 - x7) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) - 4 * (x15 - x6) * (x3 - xdes3) * (
                                   (x15 - x6) ** 2 + (x16 - x7) ** 2) - 4 * (x16 - x7) * (x4 - xdes4) * (
                                   (x15 - x6) ** 2 + (x16 - x7) ** 2) + 4 * (x14 - xdes14) * (
                                   x17 - x8) ** 3 / c ** 4 - 4 * (
                                   x17 - x8) ** 3 * (x5 - xdes5) / c ** 4


            elif i == 2:
                [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17] = np.vstack((xi, xj))
                [xdes0, xdes1, xdes2, xdes3, xdes4, xdes5, xdes6, xdes7, xdes8, xdes9, xdes10, xdes11, xdes12, xdes13,
                 xdes14, xdes15, xdes16, xdes17] = np.vstack((xi_des, xj_des))
                [uhat0, uhat1, uhat2, uhat3, uhat4, uhat5, uhat6, uhat7] = np.vstack([ui, uj])
                m1, m2 = self.get_m12()
                g1, g2 = self.get_g12()

                hdots[i] = 4 * (c ** 4 * m1 * m2 * (
                        g1 * (x0 - xdes0) * (x16 - x7) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) - g1 * (x1 - xdes1) * (
                        x15 - x6) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) + g2 * (x10 - xdes10) * (x15 - x6) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) - g2 * (x16 - x7) * (x9 - xdes9) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) + (x12 - xdes12) * (
                                2 * (x12 - xdes12) * (x15 - x6) ** 2 + (x12 - xdes12) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * (x13 - xdes13) * (x15 - x6) * (
                                        x16 - x7) + 2 * (x15 - x6) ** 2 * (-x3 + xdes3) - 2 * (x15 - x6) * (
                                        x16 - x7) * (x4 - xdes4) - (x3 - xdes3) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2)) + (x13 - xdes13) * (
                                2 * (x12 - xdes12) * (x15 - x6) * (x16 - x7) + 2 * (x13 - xdes13) * (
                                x16 - x7) ** 2 + (x13 - xdes13) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * (x15 - x6) * (x16 - x7) * (
                                        x3 - xdes3) + 2 * (x16 - x7) ** 2 * (-x4 + xdes4) - (x4 - xdes4) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2)) - (x3 - xdes3) * (
                                2 * (x12 - xdes12) * (x15 - x6) ** 2 + (x12 - xdes12) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * (x13 - xdes13) * (x15 - x6) * (
                                        x16 - x7) + 2 * (x15 - x6) ** 2 * (-x3 + xdes3) - 2 * (x15 - x6) * (
                                        x16 - x7) * (x4 - xdes4) - (x3 - xdes3) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2)) - (x4 - xdes4) * (
                                2 * (x12 - xdes12) * (x15 - x6) * (x16 - x7) + 2 * (x13 - xdes13) * (
                                x16 - x7) ** 2 + (x13 - xdes13) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * (x15 - x6) * (x16 - x7) * (
                                        x3 - xdes3) + 2 * (x16 - x7) ** 2 * (-x4 + xdes4) - (x4 - xdes4) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2))) + 3 * m1 * m2 * (
                                        x17 - x8) ** 2 * (x14 - x5 - xdes14 + xdes5) ** 2 + m1 * uhat4 * (
                                        x17 - x8) ** 3 - m2 * uhat0 * (x17 - x8) ** 3) / (c ** 4 * m1 * m2)



            elif i == 3:
                [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17] = np.vstack((xi, xj))
                [xdes0, xdes1, xdes2, xdes3, xdes4, xdes5, xdes6, xdes7, xdes8, xdes9, xdes10, xdes11, xdes12, xdes13,
                 xdes14, xdes15, xdes16, xdes17] = np.vstack((xi_des, xj_des))
                [uhat0, uhat1, uhat2, uhat3, uhat4, uhat5, uhat6, uhat7] = np.vstack([ui, uj])
                m1, m2 = self.get_m12()
                g1, g2 = self.get_g12()

                hdots[i] = 4 * (c ** 4 * m1 * m2 * (
                            g1 * uhat1 * (x16 - x7) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) - g1 * uhat2 * (x15 - x6) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) + g1 * (x0 - xdes0) * (
                                        4 * (x12 - xdes12) * (x15 - x6) * (x16 - x7) + 2 * (x13 - xdes13) * (
                                            x16 - x7) ** 2 + (x13 - xdes13) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) + (
                                                    x13 - xdes13) * ((x15 - x6) ** 2 + 3 * (x16 - x7) ** 2) - 4 * (
                                                    x15 - x6) * (x16 - x7) * (x3 - xdes3) + 2 * (x16 - x7) ** 2 * (
                                                    -x4 + xdes4) - (x4 - xdes4) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - (x4 - xdes4) * (
                                                    (x15 - x6) ** 2 + 3 * (x16 - x7) ** 2)) - g1 * (x1 - xdes1) * (
                                        2 * (x12 - xdes12) * (x15 - x6) ** 2 + (x12 - xdes12) * (
                                            (x15 - x6) ** 2 + (x16 - x7) ** 2) + (x12 - xdes12) * (
                                                    3 * (x15 - x6) ** 2 + (x16 - x7) ** 2) + 4 * (x13 - xdes13) * (
                                                    x15 - x6) * (x16 - x7) + 2 * (x15 - x6) ** 2 * (-x3 + xdes3) - 4 * (
                                                    x15 - x6) * (x16 - x7) * (x4 - xdes4) - (x3 - xdes3) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - (x3 - xdes3) * (
                                                    3 * (x15 - x6) ** 2 + (x16 - x7) ** 2)) - g2 * uhat5 * (
                                        x16 - x7) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) + g2 * uhat6 * (x15 - x6) * (
                                        (x15 - x6) ** 2 + (x16 - x7) ** 2) + g2 * (x10 - xdes10) * (
                                        2 * (x12 - xdes12) * (x15 - x6) ** 2 + (x12 - xdes12) * (
                                            (x15 - x6) ** 2 + (x16 - x7) ** 2) + (x12 - xdes12) * (
                                                    3 * (x15 - x6) ** 2 + (x16 - x7) ** 2) + 4 * (x13 - xdes13) * (
                                                    x15 - x6) * (x16 - x7) + 2 * (x15 - x6) ** 2 * (-x3 + xdes3) - 4 * (
                                                    x15 - x6) * (x16 - x7) * (x4 - xdes4) - (x3 - xdes3) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - (x3 - xdes3) * (
                                                    3 * (x15 - x6) ** 2 + (x16 - x7) ** 2)) - g2 * (x9 - xdes9) * (
                                        4 * (x12 - xdes12) * (x15 - x6) * (x16 - x7) + 2 * (x13 - xdes13) * (
                                            x16 - x7) ** 2 + (x13 - xdes13) * ((x15 - x6) ** 2 + (x16 - x7) ** 2) + (
                                                    x13 - xdes13) * ((x15 - x6) ** 2 + 3 * (x16 - x7) ** 2) - 4 * (
                                                    x15 - x6) * (x16 - x7) * (x3 - xdes3) + 2 * (x16 - x7) ** 2 * (
                                                    -x4 + xdes4) - (x4 - xdes4) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - (x4 - xdes4) * (
                                                    (x15 - x6) ** 2 + 3 * (x16 - x7) ** 2)) + (x12 - xdes12) * (
                                        2 * g1 * (x0 - xdes0) * (x15 - x6) * (x16 - x7) - 2 * g1 * (x1 - xdes1) * (
                                            x15 - x6) ** 2 - g1 * (x1 - xdes1) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * g2 * (x10 - xdes10) * (
                                                    x15 - x6) ** 2 + g2 * (x10 - xdes10) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * g2 * (x15 - x6) * (
                                                    x16 - x7) * (x9 - xdes9) + 2 * (x12 - xdes12) * (
                                                    3 * (x12 - xdes12) * (x15 - x6) + (x13 - xdes13) * (
                                                        x16 - x7) - 3 * (x15 - x6) * (x3 - xdes3) - (x16 - x7) * (
                                                                x4 - xdes4)) + 2 * (x13 - xdes13) * (
                                                    (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (x15 - x6) - (
                                                        x15 - x6) * (x4 - xdes4) - (x16 - x7) * (x3 - xdes3)) - 2 * (
                                                    x3 - xdes3) * (3 * (x12 - xdes12) * (x15 - x6) + (x13 - xdes13) * (
                                            x16 - x7) - 3 * (x15 - x6) * (x3 - xdes3) - (x16 - x7) * (
                                                                               x4 - xdes4)) - 2 * (x4 - xdes4) * (
                                                    (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (x15 - x6) - (
                                                        x15 - x6) * (x4 - xdes4) - (x16 - x7) * (x3 - xdes3))) + (
                                        x13 - xdes13) * (2 * g1 * (x0 - xdes0) * (x16 - x7) ** 2 + g1 * (x0 - xdes0) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * g1 * (x1 - xdes1) * (x15 - x6) * (
                                                                     x16 - x7) + 2 * g2 * (x10 - xdes10) * (
                                                                     x15 - x6) * (x16 - x7) - 2 * g2 * (
                                                                     x16 - x7) ** 2 * (x9 - xdes9) - g2 * (
                                                                     x9 - xdes9) * (
                                                                     (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * (
                                                                     x12 - xdes12) * (
                                                                     (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (
                                                                         x15 - x6) - (x15 - x6) * (x4 - xdes4) - (
                                                                                 x16 - x7) * (x3 - xdes3)) + 2 * (
                                                                     x13 - xdes13) * (
                                                                     (x12 - xdes12) * (x15 - x6) + 3 * (
                                                                         x13 - xdes13) * (x16 - x7) - (x15 - x6) * (
                                                                                 x3 - xdes3) - 3 * (x16 - x7) * (
                                                                                 x4 - xdes4)) - 2 * (x3 - xdes3) * (
                                                                     (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (
                                                                         x15 - x6) - (x15 - x6) * (x4 - xdes4) - (
                                                                                 x16 - x7) * (x3 - xdes3)) - 2 * (
                                                                     x4 - xdes4) * ((x12 - xdes12) * (x15 - x6) + 3 * (
                                x13 - xdes13) * (x16 - x7) - (x15 - x6) * (x3 - xdes3) - 3 * (x16 - x7) * (
                                                                                                x4 - xdes4))) - (
                                        x3 - xdes3) * (
                                        2 * g1 * (x0 - xdes0) * (x15 - x6) * (x16 - x7) - 2 * g1 * (x1 - xdes1) * (
                                            x15 - x6) ** 2 - g1 * (x1 - xdes1) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * g2 * (x10 - xdes10) * (
                                                    x15 - x6) ** 2 + g2 * (x10 - xdes10) * (
                                                    (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * g2 * (x15 - x6) * (
                                                    x16 - x7) * (x9 - xdes9) + 2 * (x12 - xdes12) * (
                                                    3 * (x12 - xdes12) * (x15 - x6) + (x13 - xdes13) * (
                                                        x16 - x7) - 3 * (x15 - x6) * (x3 - xdes3) - (x16 - x7) * (
                                                                x4 - xdes4)) + 2 * (x13 - xdes13) * (
                                                    (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (x15 - x6) - (
                                                        x15 - x6) * (x4 - xdes4) - (x16 - x7) * (x3 - xdes3)) - 2 * (
                                                    x3 - xdes3) * (3 * (x12 - xdes12) * (x15 - x6) + (x13 - xdes13) * (
                                            x16 - x7) - 3 * (x15 - x6) * (x3 - xdes3) - (x16 - x7) * (
                                                                               x4 - xdes4)) - 2 * (x4 - xdes4) * (
                                                    (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (x15 - x6) - (
                                                        x15 - x6) * (x4 - xdes4) - (x16 - x7) * (x3 - xdes3))) - (
                                        x4 - xdes4) * (2 * g1 * (x0 - xdes0) * (x16 - x7) ** 2 + g1 * (x0 - xdes0) * (
                                (x15 - x6) ** 2 + (x16 - x7) ** 2) - 2 * g1 * (x1 - xdes1) * (x15 - x6) * (
                                                                   x16 - x7) + 2 * g2 * (x10 - xdes10) * (x15 - x6) * (
                                                                   x16 - x7) - 2 * g2 * (x16 - x7) ** 2 * (
                                                                   x9 - xdes9) - g2 * (x9 - xdes9) * (
                                                                   (x15 - x6) ** 2 + (x16 - x7) ** 2) + 2 * (
                                                                   x12 - xdes12) * (
                                                                   (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (
                                                                       x15 - x6) - (x15 - x6) * (x4 - xdes4) - (
                                                                               x16 - x7) * (x3 - xdes3)) + 2 * (
                                                                   x13 - xdes13) * (
                                                                   (x12 - xdes12) * (x15 - x6) + 3 * (x13 - xdes13) * (
                                                                       x16 - x7) - (x15 - x6) * (x3 - xdes3) - 3 * (
                                                                               x16 - x7) * (x4 - xdes4)) - 2 * (
                                                                   x3 - xdes3) * (
                                                                   (x12 - xdes12) * (x16 - x7) + (x13 - xdes13) * (
                                                                       x15 - x6) - (x15 - x6) * (x4 - xdes4) - (
                                                                               x16 - x7) * (x3 - xdes3)) - 2 * (
                                                                   x4 - xdes4) * (
                                                                   (x12 - xdes12) * (x15 - x6) + 3 * (x13 - xdes13) * (
                                                                       x16 - x7) - (x15 - x6) * (x3 - xdes3) - 3 * (
                                                                               x16 - x7) * (
                                                                               x4 - xdes4)))) + 6 * m1 * uhat4 * (
                                            x17 - x8) ** 2 * (x14 - x5 - xdes14 + xdes5) - 6 * m2 * uhat0 * (
                                            x17 - x8) ** 2 * (x14 - x5 - xdes14 + xdes5) + 3 * (x17 - x8) * (
                                            2 * m1 * m2 * (x14 - x5 - xdes14 + xdes5) ** 2 + m1 * uhat4 * (
                                                x17 - x8) - m2 * uhat0 * (x17 - x8)) * (x14 - x5 - xdes14 + xdes5)) / (
                                       c ** 4 * m1 * m2)

        return hdots

    def _hdots_rect_ellipsoid(self, xdots, safety_dist):
        """Helpers for computing CBF values based on the shape of safety cage (e.g., rectangle-like or circle-like cylinders)"""
        # Get 0-th to {order-1}-th time derivatives for the ECBF constraint given "rectangular" super-ellipsoid
        assert len(xdots) >= 1 and len(xdots) <= 4
        hdots = [None] * len(xdots)
        for i in range(len(xdots)):
            if i == 0:
                x = xdots[0]
                hdots[i] = sum(x ** 4) - safety_dist ** 4
            elif i == 1:
                x, xd = xdots[0], xdots[1]
                hdots[i] = sum(4 * x ** 3 * xd)
            elif i == 2:
                x, xd, xdd = xdots[0], xdots[1], xdots[2]
                hdots[i] = sum(12 * x ** 2 * xd ** 2 + 4 * x ** 3 * xdd)
            elif i == 3:
                x, xd, xdd, xddd = xdots[0], xdots[1], xdots[2], xdots[3]
                hdots[i] = sum(24 * x * xd ** 3 + 36 * x ** 2 * xd * xdd + 4 * x ** 3 * xddd)
        return hdots

    def _control_affine_terms_rect_ellipsoid(self, xdots):
        # Get control affine terms for the ECBF constraint given "rectangular" super-ellipsoid
        # L^{order}f and LgL^{order-1}f terms
        assert len(xdots) >= 1 and len(xdots) <= 4
        x = xdots[0]  # shape: (3,)
        Lfh = 0
        LgLfh = 4 * x ** 3 * np.array([1.0, 1.0, 1.0 / self.zscale])  # shape: (3,)

        if self.order == 1:
            Lfh = 0
        elif self.order == 2:
            xd = xdots[1]
            Lfh = sum(12 * x ** 2 * xd ** 2)
        elif self.order == 3:
            xd, xdd = xdots[1], xdots[2]
            Lfh = sum(24 * x * xd ** 3 + 36 * x ** 2 * xd * xdd)
        elif self.order == 4:
            xd, xdd, xddd = xdots[1], xdots[2], xdots[3]
            Lfh = sum(24 * xd ** 4 + 144 * x * xd ** 2 * xdd + 36 * x ** 2 * xdd ** 2 + 48 * x ** 2 * xd * xddd)

        return Lfh, LgLfh

    def _hdots_circ_ellipsoid(self, xdots, safety_dist):
        # Get 0-th to {order-1}-th time derivatives for the ECBF constraint given "circular" super-ellipsoid
        assert len(xdots) >= 1 and len(xdots) <= 4
        hdots = [None] * len(xdots)
        for i in range(len(xdots)):
            if i == 0:
                x, y, z = xdots[0]  # [xi-xj, yi-yj, (zi-zj)/zscale]
                hdots[i] = (x ** 2 + y ** 2) ** 2 + z ** 4 - safety_dist ** 4
            elif i == 1:
                x, y, z = xdots[0]
                xd, yd, zd = xdots[1]
                hdots[i] = 4 * (x ** 2 + y ** 2) * (x * xd + y * yd) + 4 * z ** 3 * zd
            elif i == 2:
                x, y, z = xdots[0]
                xd, yd, zd = xdots[1]
                xdd, ydd, zdd = xdots[2]
                hdots[i] = 8 * (x * xd + y * yd) ** 2 + 4 * (x ** 2 + y ** 2) * (
                        xd ** 2 + yd ** 2 + x * xdd + y * ydd) + \
                           12 * z ** 2 * zd ** 2 + 4 * z ** 3 * zdd
            elif i == 3:
                x, y, z = xdots[0]
                xd, yd, zd = xdots[1]
                xdd, ydd, zdd = xdots[2]
                xddd, yddd, zddd = xdots[3]
                hdots[i] = 24 * (x * xd + y * yd) * (xd ** 2 + yd ** 2 + x * xdd + y * ydd) + \
                           4 * (x ** 2 + y ** 2) * (3 * xd * xdd + 3 * yd * ydd + x * xddd + y * yddd) + \
                           24 * z * zd ** 3 + 36 * z ** 2 * zd * zdd + 4 * z ** 3 * zddd
        return hdots

    def _control_affine_terms_circ_ellipsoid(self, xdots):
        # Get control affine terms for the ECBF constraint given "rectangular" super-ellipsoid
        # L^{order}f and LgL^{order-1}f terms
        assert len(xdots) >= 1 and len(xdots) <= 4
        x, y, z = xdots[0]
        Lfh = 0
        LgLfh = np.zeros((3,))
        LgLfh[0] = 4 * (x ** 2 + y ** 2) * x
        LgLfh[1] = 4 * (x ** 2 + y ** 2) * y
        LgLfh[2] = 4 * z ** 3 / self.zscale

        if self.order == 1:
            Lfh = 0

        elif self.order == 2:
            xd, yd, zd = xdots[1]
            Lfh = 8 * (x * xd + y * yd) ** 2 + 4 * (x ** 2 + y ** 2) * (xd ** 2 + yd ** 2) + 12 * z ** 2 * zd ** 2

        elif self.order == 3:
            xd, yd, zd = xdots[1]
            xdd, ydd, zdd = xdots[2]
            Lfh = 24 * (x * xd + y * yd) * (xd ** 2 + yd ** 2 + x * xdd + y * ydd) + 4 * (x ** 2 + y ** 2) * (
                    3 * xd * xdd + 3 * yd * ydd) + 24 * z * zd ** 3 + 36 * z ** 2 * zd * zdd

        elif self.order == 4:
            xd, yd, zd = xdots[1]
            xdd, ydd, zdd = xdots[2]
            xddd, yddd, zddd = xdots[3]
            Lfh = 24 * (xd ** 2 + yd ** 2 + x * xdd + y * ydd) ** 2 + 32 * (x * xd + y * yd) * (
                    3 * xd * xdd + 3 * yd * ydd + x * xddd + y * yddd) + \
                  4 * (x ** 2 + y ** 2) * (3 * xdd ** 2 + 3 * ydd ** 2 + 4 * xd * xddd + 4 * yd * yddd) + \
                  24 * zd ** 4 + 144 * z * zd ** 2 * zdd + 36 * z ** 2 * zdd ** 2 + 48 * z ** 2 * zd * zddd

        return Lfh, LgLfh

    def _build_interagent_const_ij(self, x, i, j):
        """Build Gij and hij for the ECBF constraint"""
        N = len(x)
        xi, xj = x[i], x[j]

        # Derivatives for difference in positions (0~order-1)
        # Difference in pos, vel, acc, jrk, where z axis is scaled.
        xdots = [(xi[k] - xj[k]) * np.array([1.0, 1.0, 1.0 / self.zscale]) for k in range(self.order)]

        # Compute time derivatives of barrier functions (0~order-1)
        Ds = 2 * self.safety_radius
        hdots = self._hdots(xdots, Ds)
        # Compute affine term in control input
        Lfh, LgLfh = self._ctrl_affine(xdots)

        # Build ij constraint
        Gij = np.zeros(3 * N, )
        Gij[3 * i:3 * (i + 1)] = -LgLfh
        Gij[3 * j:3 * (j + 1)] = LgLfh
        hij = np.dot(self.Kcbf, hdots) + Lfh

        return Gij, hij

    def _build_box_const(self, x, i, ignore_zmin=False, derivative_order=0):
        """Box constraint for agent i"""
        N = len(x)

        # Check if state bound exists TODO: clean up, ensure that both min & max bounds should exist in initialization
        if (self.state_bounds[derivative_order].min_bounds is None) or (
                self.state_bounds[derivative_order].max_bounds is None):
            return np.zeros((0, 3 * N)), np.zeros((0))

        xi = x[i]
        if ignore_zmin:
            Gi = np.zeros((5, 3 * N))
            hi = np.zeros((5,))
        else:
            Gi = np.zeros((6, 3 * N))
            hi = np.zeros((6,))

        # Modify default cbf poles based on the derivative ordre of box constraint (e.g., vel box constraint does not depend on position)
        Kcbf_local = self.Kcbf.copy()
        Kcbf_local[0, :derivative_order] = 0

        # Upper bounds
        Gi[0:3, 3 * i:3 * i + 3] = np.eye(3)
        xi_max = np.zeros((self.order, 3))
        xi_max[derivative_order] = self.state_bounds[derivative_order].max_bounds  # xi_max[0] = self.max_room_bounds
        hi[:3] = np.dot(Kcbf_local, xi_max - xi)

        # Lower bounds
        Gi[3, 3 * i] = -1
        Gi[4, 3 * i + 1] = -1
        if not ignore_zmin:
            Gi[5, 3 * i + 2] = -1

        xi_min = np.zeros((self.order, 3))
        xi_min[derivative_order] = self.state_bounds[derivative_order].min_bounds  # xi_min[0] = self.min_room_bounds
        if not ignore_zmin:
            hi[3:6] = np.dot(Kcbf_local, xi - xi_min)
        else:
            hi[3:5] = np.dot(Kcbf_local, xi[:, :-1] - xi_min[:, :-1])

        return Gi, hi

    """Helpers for building inequality constraints for QP"""

    def _build_ineq_const(self, x, ignore_pos_zmin, x_obs, obs_r_list):
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

        N = len(x)  # number of agents
        G = np.zeros((0, 3 * N))  # place holder
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
            G_obs, h_obs = self._build_obstacles_const(x, x_obs, obs_r_list)
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

        G = np.zeros((N_const, 3 * N))
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

    def _build_umax_const(self, N):
        # max infinity norm of control input constraint
        G = np.vstack((np.eye(3 * N), -np.eye(3 * N)))
        h = self.umax * np.ones(2 * 3 * N)
        return G, h

    def _build_state_bound_const(self, x, ignore_pos_zmin):
        # box constraint for states (including room bounds) for each agent i
        N = len(x)
        G = np.zeros((0, 3 * N))  # place holder
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
