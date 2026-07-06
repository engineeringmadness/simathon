"""Lorenz attractor physics core: ODE system + RK4 integrator.

Runs once at startup to pre-compute the full frozen 3D trail, then hands the
numpy point array off to the renderer. Integration is inherently serial (one
particle's sequential trajectory), so the kernel uses a serialized loop.
"""

import numpy as np
import taichi as ti

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01
TRANSIENT_STEPS = 500
POINT_CAP = 10000
X0, Y0, Z0 = 1.0, 1.0, 1.0


@ti.kernel
def integrate(positions: ti.types.ndarray(ndim=2, dtype=ti.f32),
              n_total: ti.i32, n_transient: ti.i32, dt: ti.f32):
    x = X0
    y = Y0
    z = Z0
    s = SIGMA
    r = RHO
    b = BETA
    ti.loop_config(serialize=True)
    for i in range(n_total + n_transient):
        k1x = s * (y - x)
        k1y = x * (r - z) - y
        k1z = x * y - b * z

        ax = x + 0.5 * dt * k1x
        ay = y + 0.5 * dt * k1y
        az = z + 0.5 * dt * k1z
        k2x = s * (ay - ax)
        k2y = ax * (r - az) - ay
        k2z = ax * ay - b * az

        bx = x + 0.5 * dt * k2x
        by = y + 0.5 * dt * k2y
        bz = z + 0.5 * dt * k2z
        k3x = s * (by - bx)
        k3y = bx * (r - bz) - by
        k3z = bx * by - b * bz

        cx = x + dt * k3x
        cy = y + dt * k3y
        cz = z + dt * k3z
        k4x = s * (cy - cx)
        k4y = cx * (r - cz) - cy
        k4z = cx * cy - b * cz

        x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

        if i >= n_transient:
            idx = i - n_transient
            positions[idx, 0] = x
            positions[idx, 1] = y
            positions[idx, 2] = z


def compute_trail():
    positions = np.zeros((POINT_CAP, 3), dtype=np.float32)
    integrate(positions, POINT_CAP, TRANSIENT_STEPS, DT)
    return positions
