"""Per-frame 3D -> 2D projection (Taichi kernel) + viridis colormap.

Each frame the projection kernel rotates the frozen trail around Y (yaw) then
X (bob/pitch), applies perspective projection, and writes screen coords plus
per-point depth used for draw ordering. The viridis colormap is a function of
each point's (constant) world z-height, so the LUT -> RGB mapping is computed
once; the two lobes sit at different heights and thus land in different hues.
"""

import numpy as np
import taichi as ti

# matplotlib viridis stops (RGB 0-1), interpolated to a 256-entry LUT.
_VIRIDIS_STOPS = [
    (0.00, (0.267004, 0.004874, 0.329415)),
    (0.10, (0.282327, 0.092911, 0.412526)),
    (0.20, (0.253935, 0.265254, 0.529983)),
    (0.30, (0.206606, 0.361778, 0.553546)),
    (0.40, (0.163625, 0.471018, 0.558831)),
    (0.50, (0.127768, 0.566949, 0.550556)),
    (0.60, (0.134692, 0.658636, 0.517649)),
    (0.70, (0.266941, 0.748751, 0.440573)),
    (0.80, (0.477504, 0.821444, 0.318195)),
    (0.90, (0.741388, 0.873449, 0.149561)),
    (1.00, (0.993248, 0.906157, 0.143936)),
]


def build_viridis_lut(size=256):
    stops = np.array([p[0] for p in _VIRIDIS_STOPS], dtype=np.float32)
    rgb = np.array([p[1] for p in _VIRIDIS_STOPS], dtype=np.float32)
    t = np.linspace(0.0, 1.0, size, dtype=np.float32)
    lut = np.empty((size, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(t, stops, rgb[:, c])
    return np.clip(lut * 255.0, 0.0, 255.0)


@ti.kernel
def project_kernel(
    trail: ti.types.ndarray(ndim=2, dtype=ti.f32),
    screen: ti.types.ndarray(ndim=2, dtype=ti.f32),
    n: ti.i32,
    yaw: ti.f32, pitch: ti.f32,
    focal: ti.f32, cam_dist: ti.f32,
    cx: ti.f32, cy: ti.f32, cz: ti.f32,
    half_w: ti.f32, half_h: ti.f32,
):
    cy_yaw = ti.cos(yaw)
    sy_yaw = ti.sin(yaw)
    cp = ti.cos(pitch)
    sp = ti.sin(pitch)
    for i in range(n):
        x = trail[i, 0] - cx
        y = trail[i, 1] - cy
        z = trail[i, 2] - cz
        x1 = x * cy_yaw + z * sy_yaw
        z1 = -x * sy_yaw + z * cy_yaw
        y1 = y
        x2 = x1
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        depth = cam_dist + z2
        sx = focal * x2 / depth
        sy = focal * y2 / depth
        screen[i, 0] = sx + half_w
        screen[i, 1] = half_h - sy
        screen[i, 2] = depth


class Renderer:
    def __init__(self, trail, width, height):
        self.trail = np.ascontiguousarray(trail, dtype=np.float32)
        self.n = self.trail.shape[0]
        self.half_w = width / 2.0
        self.half_h = height / 2.0

        z = self.trail[:, 2]
        self.z_min = float(z.min())
        self.z_max = float(z.max())
        self.z_span = max(self.z_max - self.z_min, 1e-6)

        self.center = self.trail.mean(axis=0).astype(np.float32)
        rel = self.trail - self.center
        max_r = float(np.sqrt((rel ** 2).sum(axis=1)).max())

        self.cam_dist = max_r * 3.0
        d_near = self.cam_dist - max_r
        half = min(width, height) / 2.0
        self.focal = 0.9 * half * d_near / max(max_r, 1e-6)

        self.lut = build_viridis_lut()
        zn = (self.trail[:, 2] - self.z_min) / self.z_span
        idx = np.clip((zn * 255.0).astype(np.int32), 0, 255)
        colors = self.lut[idx].astype(np.uint8)
        self.color_list = [(int(c[0]), int(c[1]), int(c[2])) for c in colors]

        self.screen = np.zeros((self.n, 3), dtype=np.float32)

    def project(self, yaw, pitch):
        project_kernel(
            self.trail, self.screen, self.n,
            yaw, pitch, self.focal, self.cam_dist,
            float(self.center[0]), float(self.center[1]), float(self.center[2]),
            self.half_w, self.half_h,
        )
        return self.screen

    def project_points(self, points3d, yaw, pitch):
        pts = np.ascontiguousarray(points3d, dtype=np.float32)
        m = pts.shape[0]
        screen = np.zeros((m, 3), dtype=np.float32)
        project_kernel(
            pts, screen, m,
            yaw, pitch, self.focal, self.cam_dist,
            float(self.center[0]), float(self.center[1]), float(self.center[2]),
            self.half_w, self.half_h,
        )
        return screen
