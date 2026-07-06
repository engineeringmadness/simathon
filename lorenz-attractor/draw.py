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


_font = None


def _get_font():
    global _font
    if _font is None:
        _font = pygame.font.Font(None, 18)
    return _font


def draw_axis(surface, screen_pts, axis):
    font = _get_font()
    aaline = pygame.draw.aaline
    for i, j, color in axis.segments:
        aaline(
            surface, color,
            (float(screen_pts[i, 0]), float(screen_pts[i, 1])),
            (float(screen_pts[j, 0]), float(screen_pts[j, 1])),
        )
    for text, idx, color in axis.labels:
        x = int(screen_pts[idx, 0]) + 4
        y = int(screen_pts[idx, 1]) - 9
        label = font.render(text, True, color)
        surface.blit(label, (x, y))


_overlay_font = None


def _get_overlay_font():
    global _overlay_font
    if _overlay_font is None:
        _overlay_font = pygame.font.Font(None, 32)
    return _overlay_font


def draw_ui(surface):
    font = _get_overlay_font()
    lines = [
        ("Lorenz Attractor", (200, 200, 200)),
        ("Drag to rotate  |  Right-click to freeze  |  Scroll to change speed", (160, 160, 160)),
    ]
    y = 8
    for text, color in lines:
        label = font.render(text, True, color)
        surface.blit(label, (8, y))
        y += 34
