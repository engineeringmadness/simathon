"""Auto-rotating cinematic camera: slow constant yaw + gentle vertical bob.

No input handling here; it just drifts. Reduced-motion mode (toggled by the
orchestrator on `R`) halves the yaw rate and freezes the bob. Pause (SPACE)
stops both.
"""

import math

YAW_RATE = math.radians(6.0)
BOB_AMPLITUDE = 0.12
BOB_FREQUENCY = 0.10


class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.bob_phase = 0.0
        self.paused = False
        self.reduced_motion = False

    def update(self, dt):
        if self.paused:
            return
        rate = YAW_RATE * (0.5 if self.reduced_motion else 1.0)
        self.yaw += rate * dt
        if not self.reduced_motion:
            self.bob_phase += 2.0 * math.pi * BOB_FREQUENCY * dt

    @property
    def pitch(self):
        if self.reduced_motion:
            return 0.0
        return BOB_AMPLITUDE * math.sin(self.bob_phase)
