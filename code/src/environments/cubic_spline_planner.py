"""
Cubic spline planner for global route representation.

Adapted from Atsushi Sakai's implementation, used in Frenet_revised project.
Provides Spline3D for converting between Frenet (s, d) and Cartesian (x, y) coordinates
along a pre-computed global highway route.

Ref:
- Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame
  (Werling et al.)
"""

import math
import bisect
import numpy as np


class Spline:
    """Cubic Spline interpolation class."""

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []
        self.x = x
        self.y = y
        self.nx = len(x)
        h = np.diff(x)

        self.a = [iy for iy in y]

        A = self._calc_A(h)
        B = self._calc_B(h)
        self.c = np.linalg.solve(A, B)

        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        if t < self.x[0] or t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

    def calcd(self, t):
        if t < self.x[0] or t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0

    def calcdd(self, t):
        if t < self.x[0] or t > self.x[-1]:
            return None
        i = self._search_index(t)
        dx = t - self.x[i]
        return 2.0 * self.c[i] + 6.0 * self.d[i] * dx

    def _search_index(self, x):
        idx = bisect.bisect(self.x, x) - 1
        return min(idx, self.nx - 2)  # Clamp to valid range

    def _calc_A(self, h):
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]
        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        return A

    def _calc_B(self, h):
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline3D:
    """3D Cubic Spline class parameterized by arc length s."""

    def __init__(self, x, y, z):
        self.s = self._calc_s(x, y, z)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)
        self.sz = Spline(self.s, z)

    def _calc_s(self, x, y, z):
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2 + idz ** 2)
                   for (idx, idy, idz) in zip(dx, dy, dz)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        return self.sx.calc(s), self.sy.calc(s), self.sz.calc(s)

    def calc_curvature(self, s):
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        return math.atan2(dy, dx)


def frenet_to_inertial(s: float, d: float, csp: Spline3D):
    """
    Convert Frenet coordinates (s, d) to Cartesian (x, y, z, yaw).

    Args:
        s: longitudinal position along the reference path
        d: lateral offset from reference path (positive = left)
        csp: Spline3D reference path

    Returns:
        (x, y, z, yaw) in world coordinates
    """
    ix, iy, iz = csp.calc_position(s)
    if ix is None or iy is None:
        raise ValueError(f"Spline position undefined at s={s}")
    iyaw = csp.calc_yaw(s)
    if iyaw is None:
        raise ValueError(f"Spline yaw undefined at s={s}")
    x = ix + d * math.cos(iyaw + math.pi / 2.0)
    y = iy + d * math.sin(iyaw + math.pi / 2.0)
    return x, y, iz, iyaw


def inertial_to_frenet_approx(x: float, y: float, csp: Spline3D,
                               s_guess: float, search_range: float = 50.0,
                               ds: float = 0.5) -> tuple:
    """
    Approximate Frenet (s, d) from Cartesian (x, y) by searching near s_guess.

    Returns:
        (s, d) approximate Frenet coordinates
    """
    s_min = max(0, s_guess - search_range)
    s_max = min(csp.s[-1], s_guess + search_range)

    best_s = s_guess
    best_dist = float('inf')

    s_test = s_min
    while s_test <= s_max:
        ix, iy, _ = csp.calc_position(s_test)
        dist = math.sqrt((x - ix) ** 2 + (y - iy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_s = s_test
        s_test += ds

    # Compute lateral offset d (signed)
    ix, iy, _ = csp.calc_position(best_s)
    yaw = csp.calc_yaw(best_s)
    # Normal vector (left-pointing)
    nx = math.cos(yaw + math.pi / 2.0)
    ny = math.sin(yaw + math.pi / 2.0)
    dx = x - ix
    dy = y - iy
    d = dx * nx + dy * ny

    return best_s, d
