#!/usr/bin/env python
from __future__ import division
import numpy as np

"""
Helper class to compute CLF inequality constraint, where G*x <= h, where x=[u^T, slack]^T.
This works for single agent.
"""

class CLF(object):

    def __init__(self, lqr, alpha_weight=None, alpha=None):
        self.lqr = lqr
        self.dim = lqr.dim
        self.order = lqr.order
        self.A = lqr.A
        self.B = lqr.B

        self.use_classK = False
        if  (alpha_weight is not None) and (alpha is not None):
            self.use_classK = True
            print("WARNING: classK function is not supported anymore.")
        self.alpha_weight = alpha_weight
        self.alpha = alpha

    def build_clf_const(self, x, xf, t, T):
        """
        Use symbolic multiplication for the CLF constraint, using dV/dx * B * u <= dV/dx * B * u^*
        """
        # Obtain specific M0, M1, M2, M4 from symbolic operation
        success, (M0, M1, M2, M3, M4) = self.get_clf_helper_matrices(T-t)
        assert success, "CLF helper matrices cannot be computed."

        # Compute G & h
        G = np.zeros((1, self.dim+1)) # +1 due to slack variable
        h = np.zeros((1, ))
        G[0, :-1] = (-xf.T.dot(M0) + x.T.dot(M1)).reshape(self.dim,)
        G[0, -1] = -1

        h[0] = sum([-xf[:,i].dot(M2).dot(xf[:,i]) +
                    2*x[:,i].dot(M3).dot(xf[:,i]) -
                    x[:,i].dot(M4).dot(x[:,i])
                    for i in range(self.dim)])

        return G, h

    def get_clf_helper_matrices(self, t):
        """Precompute symbolically the matrices used in clf constraint.
        The constraint looks like dV/dx u <= dV/dx u^* + epsilon.

        On the LHS, $\frac{\partial V(x, t)}{\partial x} B = -x_f^\top M_0 +  x^\top M_1$

        On the RHS, $\frac{\partial V(x, t)}{\partial x} B u^*(x, t) =  -x_f^\top M_2 x_f
                                                                        -x^\top M_4 x
                                                                        + 2 x^\top M_3 x_f$
        """
        success = False
        M0, M1, M2, M3, M4 = None, None, None, None, None
        if self.order==1:
            M0=np.array([[1/t]])
            M1=np.array([[1/t]])
            M2=np.array([[t**(-2)]])
            M3=np.array([[t**(-2)]])
            M4=np.array([[t**(-2)]])
            success = True
        elif self.order==2:
            M0=np.array([[6/t**2], [-2/t]])
            M1=np.array([[6/t**2], [4/t]])
            M2=np.array([[36/t**4, -12/t**3], [-12/t**3, 4/t**2]])
            M3=np.array([[36/t**4, -12/t**3], [24/t**3, -8/t**2]])
            M4=np.array([[36/t**4, 24/t**3], [24/t**3, 16/t**2]])
            success = True
        elif self.order==3:
            M0=np.array([[60/t**3], [-24/t**2], [3/t]])
            M1=np.array([[60/t**3], [36/t**2], [9/t]])
            M2=np.array([[3600/t**6, -1440/t**5, 180/t**4], [-1440/t**5, 576/t**4, -72/t**3], [180/t**4, -72/t**3, 9/t**2]])
            M3=np.array([[3600/t**6, -1440/t**5, 180/t**4], [2160/t**5, -864/t**4, 108/t**3], [540/t**4, -216/t**3, 27/t**2]])
            M4=np.array([[3600/t**6, 2160/t**5, 540/t**4], [2160/t**5, 1296/t**4, 324/t**3], [540/t**4, 324/t**3, 81/t**2]])
            success = True
        elif self.order==4:
            M0=np.array([[840/t**4], [-360/t**3], [60/t**2], [-4/t]])
            M1=np.array([[840/t**4], [480/t**3], [120/t**2], [16/t]])
            M2=np.array([[705600/t**8, -302400/t**7, 50400/t**6, -3360/t**5], [-302400/t**7, 129600/t**6, -21600/t**5, 1440/t**4], [50400/t**6, -21600/t**5, 3600/t**4, -240/t**3], [-3360/t**5, 1440/t**4, -240/t**3, 16/t**2]])
            M3=np.array([[705600/t**8, -302400/t**7, 50400/t**6, -3360/t**5], [403200/t**7, -172800/t**6, 28800/t**5, -1920/t**4], [100800/t**6, -43200/t**5, 7200/t**4, -480/t**3], [13440/t**5, -5760/t**4, 960/t**3, -64/t**2]])
            M4=np.array([[705600/t**8, 403200/t**7, 100800/t**6, 13440/t**5], [403200/t**7, 230400/t**6, 57600/t**5, 7680/t**4], [100800/t**6, 57600/t**5, 14400/t**4, 1920/t**3], [13440/t**5, 7680/t**4, 1920/t**3, 256/t**2]])
            success = True

        return success, (M0, M1, M2, M3, M4)



    """ Functions for computing CLF constraint with classK function (not used) """
    # def build_clf_const(self, x, xf, t, T):
    #     """
    #     Use class-K function alpha such that dV/dt <= -alpha_weight*alpha(V)
    #     Exponential convergence: alpha(x) = alpha_weight*x
    #     Another common choice: alpha(x) = alpha_weight*(x**3)
    #     Asymptotic convergence: alpha(x) = 0 (but this is not class K function)
    #     """
    #
    #     G = np.zeros((1, self.dim+1)) # +1 due to slack variable
    #     h = np.zeros((1, ))
    #     expAt = self.lqr.exp_At(t, T)
    #     d = xf - np.dot(expAt, x)
    #     Ginv = self.lqr.gramian_inv(t, T)
    #     dT_Ginv = d.T.dot(Ginv).reshape((self.dim, self.order)) # d.T * G
    #     LgV = np.array([-dT_Ginv[i].dot(expAt).dot(self.B)
    #                     for i in range(self.dim)]).reshape(self.dim,)
    #
    #     G[0,:-1] = LgV
    #     G[0, -1] = -1
    #
    #     if not self.use_classK:
    #         # Do not use class-K function alpha
    #         # dV/dt <= dV*/dt+slack, V* is obtained by plugging in u* from LQR in closed form.
    #         h[0] = -LgV.dot(LgV)
    #
    #     elif (self.alpha_weight is not None) and (self.alpha is not None):
    #         dGinv = self.lqr.gramian_inv_time_derivative(t, T)
    #         partial_V_t = sum(dT_Ginv[i].dot(self.A).dot(expAt).dot(x[:,i]) for i in range(self.dim)) \
    #                     + 0.5*sum(d[:,i].dot(dGinv).dot(d[:,i]) for i in range(self.dim))
    #         LfV = sum(-dT_Ginv[i].dot(expAt).dot(self.A).dot(x[:,i])
    #                         for i in range(self.dim))
    #         V = 0.5*sum(d[:,i].dot(Ginv).dot(d[:,i]) for i in range(self.dim))
    #         h[0] = -(partial_V_t+LfV) - self.alpha_weight * self.alpha(V)
    #
    #     return G, h



    # def _get_lqr_const(self, x, xf, t, T):
    #     """dV/dt <= dV*/dt+slack, V* is obtained by plugging in u* from LQR in closed form."""
    #
    #     G = np.zeros((1, self.dim+1)) # +1 due to slack variable
    #     h = np.zeros((1, ))
    #
    #     expAt = self.lqr.exp_At(t, T)
    #     d = xf - np.dot(expAt, x)
    #     Ginv = self.lqr.gramian_inv(t, T)
    #     dT_Ginv = d.T.dot(Ginv).reshape((self.dim, self.order)) # d.T * G
    #
    #     LgV = np.array([-dT_Ginv[i].dot(expAt).dot(self.B)
    #                     for i in range(self.dim)]).reshape(self.dim,)
    #     G[0,:-1] = LgV
    #     G[0, -1] = -1
    #     h[0] = -LgV.dot(LgV)
    #
    #     return G, h

#
#     def _build_CLF_asym_const(self, x, xf, t, T):
#         # LgV*u <= -LfV - partial_V_t + slack
#
#         G = np.zeros((1, self.dim+1))
#         h = np.zeros((1, ))
#
#         expAt = self.lqr.exp_At(t, T)
#         d = xf - np.dot(expAt, x)
#         Ginv = self.lqr.gramian_inv(t, T)
#         dGinv = self.lqr.gramian_inv_time_derivative(t, T)
#         dT_Ginv = d.T.dot(Ginv).reshape((self.dim, self.order)) # d.T * G
#
#         LgV = np.array([-dT_Ginv[i].dot(expAt).dot(self.B)
#                         for i in range(self.dim)]).reshape(self.dim,)
#
#         partial_V_t = sum(dT_Ginv[i].dot(self.A).dot(expAt).dot(x[:,i]) for i in range(self.dim)) \
#                         + 0.5*sum(d[:,i].dot(dGinv).dot(d[:,i]) for i in range(self.dim))
#         LfV = sum(-dT_Ginv[i].dot(expAt).dot(self.A).dot(x[:,i])
#                         for i in range(self.dim))
#
#         G[0,:-1] = LgV
#         G[0, -1] = -1
#         h[0] = -(partial_V_t+LfV)
#
#         return G, h
#
#     def _build_CLF_exp_const(self, x, xf, t, T, exp_weight=1.0):
#         # LgV*u <= -LfV - partial_V_t - exp_weight*V + slack
#
#         G = np.zeros((1, self.dim+1))
#         h = np.zeros((1, ))
#
#         expAt = self.lqr.exp_At(t, T)
#         d = xf - np.dot(expAt, x)
#         Ginv = self.lqr.gramian_inv(t, T)
#         dGinv = self.lqr.gramian_inv_time_derivative(t, T)
#         dT_Ginv = d.T.dot(Ginv).reshape((self.dim, self.order)) # d.T * G
#
#         LgV = np.array([-dT_Ginv[i].dot(expAt).dot(self.B)
#                         for i in range(self.dim)]).reshape(self.dim,)
#
#         partial_V_t = sum(dT_Ginv[i].dot(self.A).dot(expAt).dot(x[:,i]) for i in range(self.dim)) \
#                         + 0.5*sum(d[:,i].dot(dGinv).dot(d[:,i]) for i in range(self.dim))
#         LfV = sum(-dT_Ginv[i].dot(expAt).dot(self.A).dot(x[:,i])
#                         for i in range(self.dim))
#
#         G[0,:-1] = LgV
#         G[0, -1] = -1
#         h[0] = -(partial_V_t+LfV) - exp_weight*0.5*sum(d[:,i].dot(Ginv).dot(d[:,i]) for i in range(self.dim))
#
#         return G, h
#
#
#     def _build_CLF_classK_const(self, x, xf, t, T, weight=1):
#         # LgV*u <= -LfV - partial_V_t - exp_weight*V + slack
#
#         G = np.zeros((1, self.dim+1))
#         h = np.zeros((1, ))
#
#         expAt = self.lqr.exp_At(t, T)
#         d = xf - np.dot(expAt, x)
#         Ginv = self.lqr.gramian_inv(t, T)
#         dGinv = self.lqr.gramian_inv_time_derivative(t, T)
#         dT_Ginv = d.T.dot(Ginv).reshape((self.dim, self.order)) # d.T * G
#
#         LgV = np.array([-dT_Ginv[i].dot(expAt).dot(self.B)
#                         for i in range(self.dim)]).reshape(self.dim,)
#
#         partial_V_t = sum(dT_Ginv[i].dot(self.A).dot(expAt).dot(x[:,i]) for i in range(self.dim)) \
#                         + 0.5*sum(d[:,i].dot(dGinv).dot(d[:,i]) for i in range(self.dim))
#         LfV = sum(-dT_Ginv[i].dot(expAt).dot(self.A).dot(x[:,i])
#                         for i in range(self.dim))
#
#         G[0,:-1] = LgV
#         G[0, -1] = -1
#
# #         alpha = lambda v : min(v, v**2)
#         alpha = lambda v : v**3
#
#         h[0] = -(partial_V_t+LfV) - weight*alpha(0.5*sum(d[:,i].dot(Ginv).dot(d[:,i]) for i in range(self.dim)))
#
#         return G, h

