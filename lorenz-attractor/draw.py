"""Pygame rendering pass: antialiased line segments, drawn back-to-front.

Only the first `n` points of the trail are drawn, so the attractor is revealed
gradually as if a single particle traces it; once `n` reaches the point cap the
trail is frozen. Segments are sorted by average depth (distance from camera) so
nearer parts correctly overlap farther parts as the camera rotates. A small
brightened dot marks the leading particle while the trace is still growing.

Coordinates and the sort order are converted to Python lists once per frame so
the hot per-segment loop stays in pure Python (no per-element numpy indexing),
keeping the draw inside the frame budget.
"""

import numpy as np
import pygame


def draw_segments(surface, screen, color_list, n):
    if n < 2:
        return
    depths = screen[:n, 2]
    seg_depth = 0.5 * (depths[:-1] + depths[1:])
    order = np.argsort(seg_depth)[::-1].tolist()
    xs = screen[:n, 0].tolist()
    ys = screen[:n, 1].tolist()
    aaline = pygame.draw.aaline
    for s in order:
        aaline(
            surface,
            color_list[s],
            (xs[s], ys[s]),
            (xs[s + 1], ys[s + 1]),
        )


def draw_particle(surface, screen, color_list, idx):
    r, g, b = color_list[idx]
    head = (r // 2 + 128, g // 2 + 128, b // 2 + 128)
    x = int(screen[idx, 0])
    y = int(screen[idx, 1])
    pygame.draw.circle(surface, head, (x, y), 3)
