"""Linear MPC inside a safe flight corridor, solved by a small ADMM QP solver (OSQP's algorithm) in numpy.

Model: double integrator per axis, step dt, horizon N, input = acceleration (drag is compensated
downstream, in the attitude mapping). With position constraints that are axis-aligned boxes and a
diagonal cost, the three axes are independent QPs with the same matrices; they are solved together
as the three columns of one matrix problem.

Per axis, decision variables x = [a_0..a_{N-1}, s_1..s_N] (accelerations, box slacks >= 0):

    min  w_pos |p - p_ref|^2 + w_acc |a|^2 + w_jerk |a_k - a_{k-1}|^2 + w_term v_N^2
         + w_sq |s|^2 + w_lin sum(s)
    s.t. |a_k| <= a_max,  |v_k| <= v_lim_k,  lo_k - s_k <= p_k <= hi_k + s_k,  s_k >= 0

with p = p_free + Sp a and v = v_free + Sv a (condensed dynamics). Only the linear cost term and the
constraint bounds change from one call to the next, so the ADMM matrix is inverted once.
"""

import numpy as np


class CorridorMPC:
    def __init__(self, cfg):
        self.cfg = c = cfg
        n, dt = c.horizon, c.dt
        k = np.arange(1, n + 1)[:, None]
        i = np.arange(n)[None, :]
        self.Sv = dt * (i < k).astype(float)                            # v_k = v0 + sum_{i<k} a_i dt
        self.Sp = dt ** 2 * np.where(i < k, k - i - 0.5, 0.)             # p_k = p0 + k dt v0 + dt^2 sum (k-i-1/2) a_i
        self.kdt = dt * np.arange(1, n + 1)
        self.D = np.eye(n) - np.eye(n, k=-1)                              # a_k - a_{k-1}, a_{-1} = previous command
        sv_n = self.Sv[-1]

        z = np.zeros((n, n))
        eye = np.eye(n)
        p_aa = 2 * (c.w_pos * self.Sp.T @ self.Sp + c.w_acc * eye + c.w_jerk * self.D.T @ self.D
                    + c.w_term_vel * np.outer(sv_n, sv_n))
        self.P = np.block([[p_aa, z], [z, 2 * c.w_slack_quad * eye]])
        a = np.block([[eye, z],               # acceleration bounds
                      [self.Sv, z],           # velocity bounds
                      [self.Sp, eye],         # p + s >= lo
                      [self.Sp, -eye],        # p - s <= hi
                      [z, eye]])              # s >= 0
        self.row_scale = 1. / np.linalg.norm(a, axis=1)                   # unit rows: better ADMM conditioning
        self.A = a * self.row_scale[:, None]
        self.n = n
        self.rho = c.admm_rho
        self._factor()
        self.reset()

    def _factor(self):
        c = self.cfg
        k = self.P + c.admm_sigma * np.eye(self.P.shape[0]) + self.rho * self.A.T @ self.A
        self.K_inv = np.linalg.inv(k)

    def reset(self):
        n = self.n
        self.x = np.zeros((2 * n, 3))
        self.z = np.zeros((5 * n, 3))
        self.y = np.zeros((5 * n, 3))
        self.a_prev = np.zeros(3)
        self.iters = 0
        self.predicted = None

    # ------------------------------------------------------------------ public API
    def solve(self, p0, v0, p_ref, lo, hi, v_max_xy):
        """p0, v0: (3,) state; p_ref, lo, hi: (N, 3) reference and box bounds for p_1..p_N.

        Returns the first acceleration (3,). The predicted positions are kept in `self.predicted`.
        """
        c, n = self.cfg, self.n
        p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
        p_free = p0 + self.kdt[:, None] * v0
        d0 = np.zeros((n, 3))
        d0[0] = self.a_prev
        q_a = 2 * (c.w_pos * self.Sp.T @ (p_free - p_ref) - c.w_jerk * self.D.T @ d0
                   + c.w_term_vel * self.Sv[-1][:, None] * v0[None, :])
        q = np.vstack([q_a, np.full((n, 3), c.w_slack_lin)])

        a_max = np.array([c.a_max_xy_m_s2, c.a_max_xy_m_s2, c.a_max_z_m_s2])
        v_max = np.array([c.v_max_margin * v_max_xy] * 2 + [c.v_max_z_m_s])
        # Velocity limit never tighter than what can be reached from v0 (keeps the QP feasible after a fast start).
        v_lim = np.maximum(v_max, np.abs(v0) - 0.8 * a_max * self.kdt[:, None])
        inf = np.full((n, 3), np.inf)
        l = np.vstack([np.broadcast_to(-a_max, (n, 3)), -v_lim - v0, lo - p_free, -inf, np.zeros((n, 3))])
        u = np.vstack([np.broadcast_to(a_max, (n, 3)), v_lim - v0, inf, hi - p_free, inf])
        l, u = l * self.row_scale[:, None], u * self.row_scale[:, None]

        self._shift_warm_start()
        self._admm(q, l, u)

        a = self.x[:n]
        self.predicted = p_free + self.Sp @ a
        self.a_prev = a[0].copy()
        return a[0].copy()

    # ------------------------------------------------------------------ ADMM (OSQP, fixed-structure dense version)
    def _shift_warm_start(self):
        """Receding horizon: the previous solution shifted by one step is a good starting point."""
        n = self.n
        for arr, blocks in ((self.x, 2), (self.z, 5), (self.y, 5)):
            for b in range(blocks):
                seg = arr[b * n:(b + 1) * n]
                seg[:-1] = seg[1:].copy()

    def _admm(self, q, l, u):
        c = self.cfg
        P, A = self.P, self.A
        sigma, alpha = c.admm_sigma, c.admm_alpha
        x, z, y = self.x, np.clip(self.z, l, u), self.y
        it = 0
        for it in range(1, c.admm_max_iter + 1):
            x_t = self.K_inv @ (sigma * x - q + A.T @ (self.rho * z - y))
            z_t = A @ x_t
            x = alpha * x_t + (1 - alpha) * x
            z_relax = alpha * z_t + (1 - alpha) * z
            z_new = np.clip(z_relax + y / self.rho, l, u)
            y = y + self.rho * (z_relax - z_new)
            z = z_new
            if it % 10 == 0:
                ax, px, aty = A @ x, P @ x, A.T @ y
                r_prim = np.abs(ax - z).max()
                r_dual = np.abs(px + q + aty).max()
                e_prim = c.admm_eps_abs + c.admm_eps_rel * max(np.abs(ax).max(), np.abs(z).max())
                e_dual = c.admm_eps_abs + c.admm_eps_rel * max(np.abs(px).max(), np.abs(aty).max(), np.abs(q).max())
                if r_prim <= e_prim and r_dual <= e_dual:
                    break
                if it % 50 == 0:                                      # OSQP-style rho adaptation
                    ratio = np.sqrt((r_prim / max(e_prim, 1e-9)) / max(r_dual / max(e_dual, 1e-9), 1e-9))
                    new_rho = float(np.clip(self.rho * ratio, 1e-4, 1e4))
                    if not 0.2 < new_rho / self.rho < 5.:
                        self.rho = new_rho
                        self._factor()
        self.x, self.z, self.y = x, z, y
        self.iters = it
