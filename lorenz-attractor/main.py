"""Orchestrator: Taichi + Pygame init, startup integration, main loop, controls.

Wires simulation -> camera -> render -> draw each frame. Minimal controls:
ESC / window close quits, SPACE pauses camera rotation, R toggles reduced motion.
"""

import taichi as ti
import pygame

import simulation
import render as render_mod
from camera import Camera
from axis import Axis
import draw as draw_mod

WIDTH, HEIGHT = 1280, 800
FPS = 60
REVEAL_DURATION_SEC = 108.0


def main():
    ti.init(arch=ti.cpu)
    pygame.init()
    try:
        surface = pygame.display.set_mode(
            (WIDTH, HEIGHT), flags=pygame.DOUBLEBUF, vsync=1)
    except pygame.error:
        surface = pygame.display.set_mode(
            (WIDTH, HEIGHT), flags=pygame.DOUBLEBUF)
    pygame.display.set_caption("Lorenz Attractor")
    clock = pygame.time.Clock()

    trail = simulation.compute_trail()
    renderer = render_mod.Renderer(trail, WIDTH, HEIGHT)
    color_list = renderer.color_list
    camera = Camera()
    axis = Axis(trail)

    point_cap = simulation.POINT_CAP
    base_rate = point_cap / REVEAL_DURATION_SEC
    revealed = 1.0
    speed = 1.0

    frame = 0
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    camera.paused = not camera.paused
                elif event.key == pygame.K_r:
                    camera.reduced_motion = not camera.reduced_motion
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    camera.dragging = True
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    pygame.mouse.get_rel()
                elif event.button == 3:
                    camera.frozen = not camera.frozen
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    camera.dragging = False
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
            elif event.type == pygame.MOUSEMOTION:
                if camera.dragging:
                    camera.handle_mouse_drag(event.rel[0])
            elif event.type == pygame.MOUSEWHEEL:
                speed = max(0.1, min(5.0, speed + event.y * 0.1))

        camera.update(dt)
        if not camera.paused:
            revealed = revealed + base_rate * speed * dt
        trail_drawn = min(int(revealed), point_cap)
        particle_idx = int(revealed) % point_cap

        screen_pts = renderer.project(camera.yaw, camera.pitch)
        axis_pts = renderer.project_points(axis.points, camera.yaw, camera.pitch)

        surface.fill((0, 0, 0))
        draw_mod.draw_axis(surface, axis_pts, axis)
        draw_mod.draw_segments(surface, screen_pts, color_list, trail_drawn)
        draw_mod.draw_particle(surface, screen_pts, color_list, particle_idx)
        draw_mod.draw_ui(surface)
        pygame.display.flip()

        frame += 1
        if frame % 15 == 0:
            if trail_drawn < point_cap:
                pct = trail_drawn * 100 // point_cap
                pygame.display.set_caption(
                    f"Lorenz Attractor - drawing {pct}% - {speed:.1f}x - {int(clock.get_fps())} fps")
            else:
                pygame.display.set_caption(
                    f"Lorenz Attractor - {speed:.1f}x - {int(clock.get_fps())} fps")

    pygame.quit()


if __name__ == "__main__":
    main()
