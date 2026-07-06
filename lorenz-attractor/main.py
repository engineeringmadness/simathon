"""Orchestrator: Taichi + Pygame init, startup integration, main loop, controls.

Wires simulation -> camera -> render -> draw each frame. Minimal controls:
ESC / window close quits, SPACE pauses camera rotation, R toggles reduced motion.
"""

import taichi as ti
import pygame

import simulation
import render as render_mod
from camera import Camera
import draw as draw_mod

WIDTH, HEIGHT = 1280, 800
FPS = 60
REVEAL_DURATION_SEC = 36.0


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

    point_cap = simulation.POINT_CAP
    reveal_rate = point_cap / REVEAL_DURATION_SEC
    revealed = 1.0

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

        camera.update(dt)
        if not camera.paused:
            revealed = min(revealed + reveal_rate * dt, point_cap)
        n = int(revealed)

        screen_pts = renderer.project(camera.yaw, camera.pitch)

        surface.fill((0, 0, 0))
        draw_mod.draw_segments(surface, screen_pts, color_list, n)
        if n < point_cap:
            draw_mod.draw_particle(surface, screen_pts, color_list, n - 1)
        pygame.display.flip()

        frame += 1
        if frame % 15 == 0:
            if n < point_cap:
                pct = n * 100 // point_cap
                pygame.display.set_caption(
                    f"Lorenz Attractor - drawing {pct}% - {int(clock.get_fps())} fps")
            else:
                pygame.display.set_caption(
                    f"Lorenz Attractor - {int(clock.get_fps())} fps")

    pygame.quit()


if __name__ == "__main__":
    main()
