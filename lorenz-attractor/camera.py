"""Auto-rotating cinematic camera with mouse-drag manual override.

By default the camera yaws automatically. Click-drag horizontally to take
manual control; auto-rotation resumes on release. Pitch always locked to 0.
"""

import math

YAW_RATE = math.radians(6.0)
MOUSE_SENSITIVITY = 0.005


class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.paused = False
        self.reduced_motion = False
        self.dragging = False

    def update(self, dt):
        if self.paused or self.dragging:
            return
        rate = YAW_RATE * (0.5 if self.reduced_motion else 1.0)
        self.yaw += rate * dt

    def handle_mouse_drag(self, dx):
        self.yaw += dx * MOUSE_SENSITIVITY

    @property
    def pitch(self):
        return 0.0
