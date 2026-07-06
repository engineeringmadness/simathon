"""3D coordinate frame (X, Y, Z axes) framing the attractor's bounding box.

Three colored axis lines emanate from the lower corner of the trail's bounding
box, each with evenly-spaced tick marks (short perpendicular segments) and
dummy numeric labels. The geometry is world-space, so it projects and rotates
with the same camera as the trail. All 3D points are collected into one array
for a single projection pass; segments and labels reference points by index.
"""

import numpy as np

TICKS_PER_AXIS = 5
MARGIN_FRAC = 0.05
TICK_LENGTH_FRAC = 0.04

X_COLOR = (220, 90, 90)
Y_COLOR = (90, 220, 90)
Z_COLOR = (90, 140, 240)
AXIS_COLORS = [X_COLOR, Y_COLOR, Z_COLOR]
AXIS_NAMES = ["X", "Y", "Z"]


class Axis:
    def __init__(self, trail):
        lo = trail.min(axis=0).astype(np.float32)
        hi = trail.max(axis=0).astype(np.float32)
        margin = (hi - lo) * MARGIN_FRAC
        self.lo = lo - margin
        self.hi = hi + margin
        self._build()

    def _build(self):
        points = []
        segments = []
        labels = []
        origin = self.lo.copy()
        extents = self.hi - self.lo

        for ax in range(3):
            tip = origin.copy()
            tip[ax] = self.hi[ax]
            color = AXIS_COLORS[ax]

            i0 = len(points)
            points.append(origin.copy())
            i1 = len(points)
            points.append(tip.copy())
            segments.append((i0, i1, color))
            labels.append((AXIS_NAMES[ax], i1, color))

            tick_len = float(extents[ax] * TICK_LENGTH_FRAC)
            perp = np.zeros(3, dtype=np.float32)
            perp[1 if ax == 0 else 0] = 1.0
            for t in range(1, TICKS_PER_AXIS + 1):
                frac = t / TICKS_PER_AXIS
                p = origin.copy()
                p[ax] = origin[ax] + extents[ax] * frac
                ia = len(points)
                points.append(p.copy())
                ib = len(points)
                points.append(p + perp * tick_len)
                segments.append((ia, ib, color))
                val = int(round(float(origin[ax] + extents[ax] * frac)))
                labels.append((str(val), ia, color))

        self.points = np.array(points, dtype=np.float32)
        self.segments = segments
        self.labels = labels
